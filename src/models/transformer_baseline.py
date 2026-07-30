from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.od_tensor_tokenizer import ODTensorTokenizer


class ODTensorTransformer(nn.Module):
    """Minimal OD tensor tokenizer + Transformer encoder baseline.

    Input:
        x: [B, L, N, N]

    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        rank: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_tokens: int = 4096,
        decoder_mode: str = "low_rank",
        decoder_scale: float = 1.0,
        zero_init_decoder: bool = False,
        use_pair_trend_head: bool = False,
        pair_trend_mode: str = "shared",
        pair_trend_scale: float = 1.0,
        zero_init_pair_trend: bool = True,
        residual_activation: str = "relu",
        softplus_beta: float = 10.0,
        context_pooling: str = "mean",
        horizon_attention_heads: int | None = None,
        poi_features: torch.Tensor | None = None,
        poi_feature_dim: int = 0,
        use_poi_features: bool = False,
        poi_projection_scale: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.rank = rank
        self.d_model = d_model
        self.decoder_mode = decoder_mode
        self.decoder_scale = decoder_scale
        self.use_pair_trend_head = use_pair_trend_head
        self.pair_trend_mode = pair_trend_mode
        self.pair_trend_scale = pair_trend_scale
        self.residual_activation = residual_activation
        self.softplus_beta = softplus_beta
        self.context_pooling = context_pooling

        token_count = input_len * rank * rank
        if token_count > max_tokens:
            raise ValueError(f"Token count {token_count} exceeds max_tokens={max_tokens}")

        self.tokenizer = ODTensorTokenizer(
            num_nodes=num_nodes,
            rank=rank,
            d_model=d_model,
            max_input_len=input_len,
            dropout=dropout,
            poi_features=poi_features,
            poi_feature_dim=poi_feature_dim,
            use_poi_features=use_poi_features,
            poi_projection_scale=poi_projection_scale,
        )
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_norm = nn.LayerNorm(d_model)
        self.horizon_embedding = nn.Embedding(pred_len, d_model)
        if context_pooling == "horizon_attention":
            attn_heads = int(horizon_attention_heads or n_heads)
            if d_model % attn_heads != 0:
                raise ValueError(f"d_model={d_model} must be divisible by horizon_attention_heads={attn_heads}")
            self.horizon_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
        elif context_pooling == "mean":
            self.horizon_attention = None
        else:
            raise ValueError(f"Unsupported context_pooling: {context_pooling}")
        self.core_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, rank * rank),
        )
        if zero_init_decoder:
            last = self.core_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if use_pair_trend_head:
            if pair_trend_mode == "shared":
                self.pair_trend_head = nn.Linear(input_len, pred_len)
                self.pair_trend_weight = None
                self.pair_trend_bias = None
                if zero_init_pair_trend:
                    nn.init.zeros_(self.pair_trend_head.weight)
                    nn.init.zeros_(self.pair_trend_head.bias)
            elif pair_trend_mode == "pair_specific":
                self.pair_trend_head = None
                self.pair_trend_weight = nn.Parameter(torch.empty(num_nodes, num_nodes, input_len, pred_len))
                self.pair_trend_bias = nn.Parameter(torch.empty(pred_len, num_nodes, num_nodes))
                if zero_init_pair_trend:
                    nn.init.zeros_(self.pair_trend_weight)
                    nn.init.zeros_(self.pair_trend_bias)
                else:
                    nn.init.normal_(self.pair_trend_weight, std=0.01)
                    nn.init.zeros_(self.pair_trend_bias)
            else:
                raise ValueError(f"Unsupported pair_trend_mode: {pair_trend_mode}")
        else:
            self.pair_trend_head = None
            self.pair_trend_weight = None
            self.pair_trend_bias = None

    def _activate_nonnegative(self, raw: torch.Tensor) -> torch.Tensor:
        if self.residual_activation == "relu":
            return F.relu(raw)
        if self.residual_activation in {"softplus_shift", "shifted_softplus"}:
            # Keeps f(0)=0 while retaining a useful positive gradient at zero-flow OD pairs.
            zero = torch.zeros((), device=raw.device, dtype=raw.dtype)
            offset = F.softplus(zero, beta=self.softplus_beta)
            return (F.softplus(raw, beta=self.softplus_beta) - offset).clamp_min(0.0)
        if self.residual_activation == "softplus":
            return F.softplus(raw, beta=self.softplus_beta)
        raise ValueError(f"Unsupported residual_activation: {self.residual_activation}")

    def _decode(
        self,
        x: torch.Tensor,
        core_pred: torch.Tensor,
        poi_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        od_delta = self.tokenizer.reconstruct(core_pred, poi_features=poi_features)
        if self.use_pair_trend_head:
            if self.pair_trend_mode == "shared":
                # Shared pair-local trend: [B, L, N, N] -> [B, H, N, N].
                pair_history = x.permute(0, 2, 3, 1)
                pair_delta = self.pair_trend_head(pair_history).permute(0, 3, 1, 2)
            elif self.pair_trend_mode == "pair_specific":
                # OD-pair-specific temporal residual: x [B,L,N,N], W [N,N,L,H] -> [B,H,N,N].
                pair_delta = torch.einsum("blij,ijlh->bhij", x, self.pair_trend_weight)
                pair_delta = pair_delta + self.pair_trend_bias.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported pair_trend_mode: {self.pair_trend_mode}")
            od_delta = od_delta + self.pair_trend_scale * pair_delta
        if self.decoder_mode == "low_rank":
            return F.softplus(od_delta)
        if self.decoder_mode == "residual_last":
            base = x[:, -1:].expand(-1, self.pred_len, -1, -1)
            return self._activate_nonnegative(base + self.decoder_scale * od_delta)
        if self.decoder_mode == "residual_mean":
            base = x.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1, -1)
            return self._activate_nonnegative(base + self.decoder_scale * od_delta)
        raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")

    def forward(
        self,
        x: torch.Tensor,
        poi_features: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        tokens, history_core = self.tokenizer(x, poi_features=poi_features)  # tokens: [B, L*r*r, d_model]
        token_ids = torch.arange(tokens.shape[1], device=x.device)
        tokens = tokens + self.position_embedding(token_ids).unsqueeze(0)

        encoded = self.encoder(tokens)
        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.context_pooling == "mean":
            context = self.context_norm(encoded.mean(dim=1))  # [B, d_model]
            future_context = context.unsqueeze(1) + horizon
        elif self.context_pooling == "horizon_attention":
            # Horizon-specific queries attend to history OD tokens:
            # query [B,H,D], key/value [B,L*r*r,D] -> future_context [B,H,D].
            attended, _ = self.horizon_attention(horizon, encoded, encoded, need_weights=False)
            future_context = self.context_norm(horizon + attended)
        else:
            raise ValueError(f"Unsupported context_pooling: {self.context_pooling}")

        core_flat = self.core_head(future_context)
        core_pred = core_flat.reshape(x.shape[0], self.pred_len, self.rank, self.rank)
        y_pred = self._decode(x, core_pred, poi_features=poi_features)

        if return_latent:
            return {
                "prediction": y_pred,
                "history_core": history_core,
                "future_core": core_pred,
                "tokens": tokens,
                "encoded_tokens": encoded,
            }
        return y_pred
