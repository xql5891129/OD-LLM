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
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.rank = rank
        self.d_model = d_model

        token_count = input_len * rank * rank
        if token_count > max_tokens:
            raise ValueError(f"Token count {token_count} exceeds max_tokens={max_tokens}")

        self.tokenizer = ODTensorTokenizer(
            num_nodes=num_nodes,
            rank=rank,
            d_model=d_model,
            max_input_len=input_len,
            dropout=dropout,
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
        self.core_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, rank * rank),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        tokens, history_core = self.tokenizer(x)  # tokens: [B, L*r*r, d_model]
        token_ids = torch.arange(tokens.shape[1], device=x.device)
        tokens = tokens + self.position_embedding(token_ids).unsqueeze(0)

        encoded = self.encoder(tokens)
        context = self.context_norm(encoded.mean(dim=1))  # [B, d_model]

        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon = self.horizon_embedding(horizon_ids).unsqueeze(0)
        future_context = context.unsqueeze(1) + horizon

        core_flat = self.core_head(future_context)
        core_pred = core_flat.reshape(x.shape[0], self.pred_len, self.rank, self.rank)
        od_logits = self.tokenizer.reconstruct(core_pred)
        y_pred = F.softplus(od_logits)

        if return_latent:
            return {
                "prediction": y_pred,
                "history_core": history_core,
                "future_core": core_pred,
                "tokens": tokens,
                "encoded_tokens": encoded,
            }
        return y_pred

