from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_blocks import MiniGPTBackbone, ReprogrammingLayer
from models.odmixer_baseline import ODMixerBaseline
from models.od_tensor_tokenizer import ODTensorTokenizer


class ODLLM(nn.Module):
    """OD tensor tokenizer + LLM-style backbone.

    Input:
        x: [B, L, N, N]

    Output:
        y_pred: [B, H, N, N]

    Two runtime modes are supported:
        1. pretrained=True: load a HuggingFace AutoModel, e.g. Qwen or DeepSeek.
        2. pretrained=False: use MiniGPTBackbone for offline smoke tests.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        rank: int = 4,
        d_model: int = 64,
        llm_dim: int = 768,
        llm_layers: int = 6,
        llm_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_tokens: int = 4096,
        llm_model: str = "mini_gpt",
        pretrained: bool = False,
        pretrained_path: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: str | None = "auto",
        attn_implementation: str | None = None,
        freeze_llm: bool = True,
        llm_trainable_layers: int = 0,
        llm_trainable_layer_norm: bool = False,
        llm_trainable_fp32: bool = True,
        llm_gradient_checkpointing: bool = False,
        use_reprogramming: bool = False,
        num_virtual_prompt_tokens: int = 8,
        num_source_tokens: int = 1000,
        decoder_mode: str = "low_rank",
        decoder_scale: float = 1.0,
        zero_init_decoder: bool = False,
        use_pair_trend_head: bool = False,
        pair_trend_mode: str = "shared",
        pair_trend_scale: float = 1.0,
        zero_init_pair_trend: bool = True,
        residual_activation: str = "relu",
        softplus_beta: float = 10.0,
        seasonal_blend_init: float = 0.7,
        learnable_seasonal_blend: bool = True,
        context_pooling: str = "mean",
        horizon_attention_heads: int | None = None,
        use_time_features: bool = False,
        time_feature_dim: int = 0,
        poi_features: torch.Tensor | None = None,
        poi_feature_dim: int = 0,
        use_poi_features: bool = False,
        poi_projection_scale: float = 0.1,
        token_value_transform: str = "none",
        projection_mode: str = "learnable",
        projection_init: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_limit: float = 0.0,
        use_context_base_gate: bool = False,
        context_base_gate_init: float | None = None,
        pair_hidden_dim: int = 64,
        pair_input_transform: str = "log1p",
        use_decoder_pair_gate: bool = False,
        decoder_pair_gate_init: float = 0.5,
        use_full_rank_pair_decoder: bool = False,
        full_rank_hidden_dim: int = 64,
        full_rank_base_gate_init: float = 0.8,
        full_rank_delta_scale: float = 1.0,
        full_rank_delta_limit: float = 8.0,
        use_marginal_features: bool = False,
        use_relative_pair_delta: bool = False,
        relative_pair_delta_limit: float = 1.0,
        use_sparse_hurdle_head: bool = False,
        occurrence_init: float = 0.2,
        occurrence_prior_scale: float = 1.0,
        occurrence_temperature: float = 1.0,
        output_calibration_scale: float | list[float] = 1.0,
        occurrence_calibration_bias: float | list[float] = 0.0,
        occurrence_calibration_temperature: float | list[float] = 1.0,
        occurrence_probability_threshold: float | list[float] = 0.0,
        prediction_calibration_power: float | list[float] = 1.0,
        prediction_value_threshold: float | list[float] = 0.0,
        evidence_calibration_weights: list[list[float]] | None = None,
        evidence_calibration_threshold: float | list[float] = 0.0,
        use_local_od_mixer: bool = False,
        local_mixer_hidden_dim: int = 32,
        local_mixer_layers: int = 5,
        local_mixer_softplus_beta: float = 10.0,
        freeze_local_od_mixer: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.rank = rank
        self.d_model = d_model
        self.llm_dim = llm_dim
        self.use_reprogramming = use_reprogramming
        self.num_virtual_prompt_tokens = num_virtual_prompt_tokens
        self.num_source_tokens = num_source_tokens
        self.decoder_mode = decoder_mode
        self.decoder_scale = decoder_scale
        self.use_pair_trend_head = use_pair_trend_head
        self.pair_trend_mode = pair_trend_mode
        self.pair_trend_scale = pair_trend_scale
        self.residual_activation = residual_activation
        self.softplus_beta = softplus_beta
        self.seasonal_blend_init = seasonal_blend_init
        self.learnable_seasonal_blend = learnable_seasonal_blend
        self.context_pooling = context_pooling
        self.use_time_features = use_time_features
        self.time_feature_dim = time_feature_dim
        self.delta_limit = float(delta_limit)
        self.use_context_base_gate = use_context_base_gate
        self.pair_input_transform = pair_input_transform
        self.use_decoder_pair_gate = use_decoder_pair_gate
        self.use_full_rank_pair_decoder = use_full_rank_pair_decoder
        self.full_rank_delta_scale = float(full_rank_delta_scale)
        self.full_rank_delta_limit = float(full_rank_delta_limit)
        self.use_marginal_features = bool(use_marginal_features)
        self.use_relative_pair_delta = bool(use_relative_pair_delta)
        self.relative_pair_delta_limit = float(relative_pair_delta_limit)
        self.use_sparse_hurdle_head = bool(use_sparse_hurdle_head)
        self.occurrence_prior_scale = float(occurrence_prior_scale)
        self.occurrence_temperature = max(float(occurrence_temperature), 1e-4)
        self.output_calibration_scale = output_calibration_scale
        self.occurrence_calibration_bias = occurrence_calibration_bias
        self.occurrence_calibration_temperature = occurrence_calibration_temperature
        self.occurrence_probability_threshold = occurrence_probability_threshold
        self.prediction_calibration_power = prediction_calibration_power
        self.prediction_value_threshold = prediction_value_threshold
        self.evidence_calibration_weights = evidence_calibration_weights
        self.evidence_calibration_threshold = evidence_calibration_threshold
        self.use_local_od_mixer = bool(use_local_od_mixer)

        token_count = input_len * rank * rank
        full_seq_len = token_count + num_virtual_prompt_tokens
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
            value_transform=token_value_transform,
            projection_mode=projection_mode,
            projection_init=projection_init,
        )
        if self.use_local_od_mixer:
            self.local_od_mixer = ODMixerBaseline(
                num_nodes=num_nodes,
                input_len=input_len,
                pred_len=pred_len,
                hidden_dim=local_mixer_hidden_dim,
                layer_nums=local_mixer_layers,
                dropout=dropout,
                prev_od_mode="lag1",
                output_activation="softplus_shift",
                softplus_beta=local_mixer_softplus_beta,
            )
            if freeze_local_od_mixer:
                for parameter in self.local_od_mixer.parameters():
                    parameter.requires_grad = False
        else:
            self.local_od_mixer = None
        self.llm_model, real_llm_dim, source_embeddings = self._build_backbone(
            llm_model=llm_model,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            llm_dim=llm_dim,
            llm_layers=llm_layers,
            llm_heads=llm_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=full_seq_len,
            num_source_tokens=num_source_tokens,
        )
        self.llm_dim = real_llm_dim
        self.llm_dtype = self._infer_model_dtype(self.llm_model)

        if llm_gradient_checkpointing and hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self.llm_model.gradient_checkpointing_enable()

        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            self._unfreeze_llm_tail(
                trainable_layers=llm_trainable_layers,
                trainable_layer_norm=llm_trainable_layer_norm,
            )
            if llm_trainable_fp32:
                self._cast_trainable_llm_params_to_fp32()

        if use_reprogramming and source_embeddings is not None:
            d_keys = max(8, d_model // max(llm_heads, 1))
            self.input_adapter = ReprogrammingLayer(d_model, llm_heads, d_keys, self.llm_dim, dropout)
            self.register_buffer("source_embeddings", source_embeddings, persistent=False)
        else:
            self.input_adapter = nn.Linear(d_model, self.llm_dim)
            self.source_embeddings = None

        if use_time_features:
            if time_feature_dim <= 0:
                raise ValueError("use_time_features=true requires a positive time_feature_dim.")
            self.history_time_proj = nn.Sequential(
                nn.Linear(time_feature_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.llm_dim),
            )
            self.future_time_proj = nn.Sequential(
                nn.Linear(time_feature_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.llm_dim),
            )
        else:
            self.history_time_proj = None
            self.future_time_proj = None

        self.virtual_prompt = nn.Parameter(torch.randn(num_virtual_prompt_tokens, self.llm_dim) * 0.02)
        self.context_norm = nn.LayerNorm(self.llm_dim)
        self.horizon_embedding = nn.Embedding(pred_len, self.llm_dim)
        init = min(max(float(seasonal_blend_init), 1e-4), 1.0 - 1e-4)
        blend_logit = torch.logit(torch.full((pred_len, 1, 1), init))
        if learnable_seasonal_blend:
            self.seasonal_blend_logit = nn.Parameter(blend_logit)
        else:
            self.register_buffer("seasonal_blend_logit", blend_logit, persistent=True)
        if use_context_base_gate:
            self.context_base_gate_head = nn.Linear(self.llm_dim, 1)
            nn.init.zeros_(self.context_base_gate_head.weight)
            gate_init = seasonal_blend_init if context_base_gate_init is None else float(context_base_gate_init)
            gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.context_base_gate_head.bias, float(torch.logit(torch.tensor(gate_init))))
        else:
            self.context_base_gate_head = None
        if context_pooling == "horizon_attention":
            attn_heads = int(horizon_attention_heads or llm_heads)
            if self.llm_dim % attn_heads != 0:
                raise ValueError(f"llm_dim={self.llm_dim} must be divisible by horizon_attention_heads={attn_heads}")
            self.horizon_attention = nn.MultiheadAttention(
                embed_dim=self.llm_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
        elif context_pooling == "mean":
            self.horizon_attention = None
        else:
            raise ValueError(f"Unsupported context_pooling: {context_pooling}")
        self.core_head = nn.Sequential(
            nn.Linear(self.llm_dim, dim_feedforward),
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
            elif pair_trend_mode == "seasonal_mlp":
                self.pair_trend_head = nn.Sequential(
                    nn.Linear(input_len + pred_len, pair_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(pair_hidden_dim, pred_len),
                )
                self.pair_trend_weight = None
                self.pair_trend_bias = None
                if zero_init_pair_trend:
                    nn.init.zeros_(self.pair_trend_head[-1].weight)
                    nn.init.zeros_(self.pair_trend_head[-1].bias)
            else:
                raise ValueError(f"Unsupported pair_trend_mode: {pair_trend_mode}")
        else:
            self.pair_trend_head = None
            self.pair_trend_weight = None
            self.pair_trend_bias = None
        if use_decoder_pair_gate:
            self.decoder_pair_gate_head = nn.Linear(self.llm_dim, 1)
            nn.init.zeros_(self.decoder_pair_gate_head.weight)
            gate_init = min(max(float(decoder_pair_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(
                self.decoder_pair_gate_head.bias,
                float(torch.logit(torch.tensor(gate_init))),
            )
        else:
            self.decoder_pair_gate_head = None
        if use_full_rank_pair_decoder:
            if decoder_mode not in {"residual_seasonal", "seasonal_residual"}:
                raise ValueError("Full-rank pair decoder requires decoder_mode=residual_seasonal.")
            self.pair_local_encoder = nn.Sequential(
                nn.Linear(input_len + pred_len, full_rank_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(full_rank_hidden_dim),
            )
            self.pair_context_proj = nn.Linear(self.llm_dim, full_rank_hidden_dim)
            self.pair_origin_embedding = nn.Embedding(num_nodes, full_rank_hidden_dim)
            self.pair_dest_embedding = nn.Embedding(num_nodes, full_rank_hidden_dim)
            self.full_rank_pair_fusion = nn.Sequential(
                nn.LayerNorm(full_rank_hidden_dim),
                nn.Linear(full_rank_hidden_dim, full_rank_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.full_rank_base_gate_head = nn.Linear(full_rank_hidden_dim, 1)
            self.full_rank_delta_head = nn.Linear(full_rank_hidden_dim, 1)
            if self.use_local_od_mixer:
                self.local_global_gate_head = nn.Linear(full_rank_hidden_dim, 1)
                nn.init.zeros_(self.local_global_gate_head.weight)
                nn.init.zeros_(self.local_global_gate_head.bias)
            else:
                self.local_global_gate_head = None
            gate_init = min(max(float(full_rank_base_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.normal_(self.full_rank_base_gate_head.weight, std=0.01)
            nn.init.constant_(
                self.full_rank_base_gate_head.bias,
                float(torch.logit(torch.tensor(gate_init))),
            )
            nn.init.zeros_(self.full_rank_delta_head.weight)
            nn.init.zeros_(self.full_rank_delta_head.bias)
            if self.use_marginal_features:
                self.origin_marginal_encoder = nn.Sequential(
                    nn.Linear(input_len + 1, full_rank_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(full_rank_hidden_dim),
                )
                self.dest_marginal_encoder = nn.Sequential(
                    nn.Linear(input_len + 1, full_rank_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(full_rank_hidden_dim),
                )
            else:
                self.origin_marginal_encoder = None
                self.dest_marginal_encoder = None
            if self.use_relative_pair_delta:
                self.full_rank_relative_head = nn.Linear(full_rank_hidden_dim, 1)
                nn.init.zeros_(self.full_rank_relative_head.weight)
                nn.init.zeros_(self.full_rank_relative_head.bias)
            else:
                self.full_rank_relative_head = None
            if self.use_sparse_hurdle_head:
                occurrence_init = min(max(float(occurrence_init), 1e-4), 1.0 - 1e-4)
                self.full_rank_occurrence_head = nn.Linear(full_rank_hidden_dim, 1)
                nn.init.normal_(self.full_rank_occurrence_head.weight, std=0.01)
                nn.init.constant_(
                    self.full_rank_occurrence_head.bias,
                    float(torch.logit(torch.tensor(occurrence_init))),
                )
            else:
                self.full_rank_occurrence_head = None
            if use_poi_features and poi_feature_dim > 0:
                self.pair_poi_norm = nn.LayerNorm(poi_feature_dim)
                self.pair_poi_origin_proj = nn.Linear(poi_feature_dim, full_rank_hidden_dim)
                self.pair_poi_dest_proj = nn.Linear(poi_feature_dim, full_rank_hidden_dim)
            else:
                self.pair_poi_norm = None
                self.pair_poi_origin_proj = None
                self.pair_poi_dest_proj = None
        else:
            self.pair_local_encoder = None
            self.pair_context_proj = None
            self.pair_origin_embedding = None
            self.pair_dest_embedding = None
            self.full_rank_pair_fusion = None
            self.full_rank_base_gate_head = None
            self.full_rank_delta_head = None
            self.local_global_gate_head = None
            self.origin_marginal_encoder = None
            self.dest_marginal_encoder = None
            self.full_rank_relative_head = None
            self.full_rank_occurrence_head = None
            self.pair_poi_norm = None
            self.pair_poi_origin_proj = None
            self.pair_poi_dest_proj = None

    def _build_backbone(
        self,
        llm_model: str,
        pretrained: bool,
        pretrained_path: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: str | None,
        attn_implementation: str | None,
        llm_dim: int,
        llm_layers: int,
        llm_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        num_source_tokens: int,
    ):
        if not pretrained:
            backbone = MiniGPTBackbone(
                hidden_size=llm_dim,
                num_layers=llm_layers,
                num_heads=llm_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            return backbone, llm_dim, None

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for pretrained HuggingFace LLM backbones. "
                "Use model.pretrained=false for offline smoke tests."
            ) from exc

        model_name_or_path = pretrained_path or llm_model
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self._try_set_num_layers(config, llm_layers)
        config.output_hidden_states = False
        config.output_attentions = False
        dtype_arg = self._resolve_torch_dtype(torch_dtype)
        model_kwargs = {
            "config": config,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if dtype_arg is not None:
            model_kwargs["torch_dtype"] = dtype_arg
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            backbone = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        except TypeError:
            if "torch_dtype" not in model_kwargs:
                raise
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            backbone = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        hidden_size = self._infer_hidden_size(config, backbone)
        source_embeddings = backbone.get_input_embeddings().weight[:num_source_tokens].detach().float()
        return backbone, hidden_size, source_embeddings

    @staticmethod
    def _try_set_num_layers(config, llm_layers: int) -> None:
        """Best-effort layer truncation across common HF config names."""
        for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
            if hasattr(config, attr):
                setattr(config, attr, llm_layers)
        for nested_attr in ["text_config", "language_config"]:
            nested = getattr(config, nested_attr, None)
            if nested is None:
                continue
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(nested, attr):
                    setattr(nested, attr, llm_layers)

    @staticmethod
    def _infer_hidden_size(config, model: nn.Module) -> int:
        for attr in ["hidden_size", "n_embd", "d_model"]:
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        for nested_attr in ["text_config", "language_config"]:
            nested = getattr(config, nested_attr, None)
            if nested is None:
                continue
            for attr in ["hidden_size", "n_embd", "d_model"]:
                value = getattr(nested, attr, None)
                if value is not None:
                    return int(value)
        emb = model.get_input_embeddings()
        return int(emb.embedding_dim)

    @staticmethod
    def _infer_model_dtype(model: nn.Module) -> torch.dtype:
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @staticmethod
    def _get_module_by_path(root: nn.Module, path: str):
        module = root
        for attr in path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                return None
        return module

    def _find_llm_layers(self) -> list[nn.Module]:
        """Find transformer block lists across HuggingFace and MiniGPT backbones."""
        candidates = [
            "layers",
            "model.layers",
            "language_model.layers",
            "language_model.model.layers",
            "model.language_model.layers",
            "model.language_model.model.layers",
            "text_model.layers",
            "text_model.model.layers",
            "model.text_model.layers",
            "model.text_model.model.layers",
            "encoder.layers",
            "encoder.layer",
            "transformer.h",
            "h",
        ]
        for path in candidates:
            module = self._get_module_by_path(self.llm_model, path)
            if isinstance(module, (nn.ModuleList, list, tuple)) and len(module) > 0:
                return list(module)
        return []

    def _unfreeze_llm_tail(self, trainable_layers: int, trainable_layer_norm: bool) -> None:
        if trainable_layers > 0:
            layers = self._find_llm_layers()
            if not layers:
                raise ValueError("llm_trainable_layers was set, but no transformer layers were found.")
            for layer in layers[-int(trainable_layers) :]:
                for param in layer.parameters():
                    param.requires_grad = True
        if trainable_layer_norm:
            for name, param in self.llm_model.named_parameters():
                lowered = name.lower()
                if "norm" in lowered or "layernorm" in lowered or "ln_" in lowered:
                    param.requires_grad = True

    def _cast_trainable_llm_params_to_fp32(self) -> None:
        for param in self.llm_model.parameters():
            if param.requires_grad and param.dtype in {torch.float16, torch.bfloat16}:
                param.data = param.data.float()

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str | None):
        if dtype_name is None:
            return None
        value = str(dtype_name).lower()
        if value in {"none", "null", "false"}:
            return None
        if value == "auto":
            return "auto"
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        if value in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

    def _adapt_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.input_adapter, ReprogrammingLayer):
            return self.input_adapter(tokens, self.source_embeddings)
        return self.input_adapter(tokens)

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

    def _transform_pair_history(self, value: torch.Tensor) -> torch.Tensor:
        if self.pair_input_transform == "none":
            return value
        if self.pair_input_transform == "log1p":
            return torch.log1p(value.clamp_min(0.0))
        if self.pair_input_transform in {"signed_log1p", "symlog"}:
            return value.sign() * torch.log1p(value.abs())
        raise ValueError(f"Unsupported pair_input_transform: {self.pair_input_transform}")

    def _horizon_parameter(
        self,
        value: float | list[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if tensor.numel() == 1:
            tensor = tensor.expand(self.pred_len)
        if tensor.numel() != self.pred_len:
            raise ValueError(f"Expected one value or {self.pred_len} horizon values, got {tensor.numel()}")
        return tensor.view(1, self.pred_len, 1, 1)

    def _decode(
        self,
        x: torch.Tensor,
        core_pred: torch.Tensor,
        poi_features: torch.Tensor | None = None,
        prev_y: torch.Tensor | None = None,
        prev_valid: torch.Tensor | None = None,
        base_blend: torch.Tensor | None = None,
        pair_gate: torch.Tensor | None = None,
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
            elif self.pair_trend_mode == "seasonal_mlp":
                seasonal = (
                    prev_y.to(device=x.device, dtype=x.dtype)
                    if prev_y is not None
                    else x[:, -1:].expand(-1, self.pred_len, -1, -1)
                )
                pair_features = torch.cat(
                    [
                        self._transform_pair_history(x).permute(0, 2, 3, 1),
                        self._transform_pair_history(seasonal).permute(0, 2, 3, 1),
                    ],
                    dim=-1,
                )
                pair_delta = self.pair_trend_head(pair_features).permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported pair_trend_mode: {self.pair_trend_mode}")
            if pair_gate is not None:
                pair_delta = pair_gate.to(device=x.device, dtype=x.dtype) * pair_delta
            od_delta = od_delta + self.pair_trend_scale * pair_delta
        if self.delta_limit > 0:
            od_delta = self.delta_limit * torch.tanh(od_delta / self.delta_limit)
        if self.decoder_mode == "low_rank":
            return F.softplus(od_delta)
        if self.decoder_mode in {"signed_low_rank", "delta", "residual_delta"}:
            return self.decoder_scale * od_delta
        if self.decoder_mode == "residual_last":
            base = x[:, -1:].expand(-1, self.pred_len, -1, -1)
            return self._activate_nonnegative(base + self.decoder_scale * od_delta)
        if self.decoder_mode in {"residual_seasonal", "seasonal_residual"}:
            last_base = x[:, -1:].expand(-1, self.pred_len, -1, -1)
            if prev_y is None:
                base = last_base
            else:
                seasonal = prev_y.to(device=x.device, dtype=x.dtype)
                if base_blend is None:
                    blend = torch.sigmoid(self.seasonal_blend_logit).to(device=x.device, dtype=x.dtype)
                    blend = blend.unsqueeze(0)
                else:
                    blend = base_blend.to(device=x.device, dtype=x.dtype)
                base = blend * seasonal + (1.0 - blend) * last_base
                if prev_valid is not None:
                    valid = prev_valid.to(device=x.device, dtype=torch.bool).view(-1, 1, 1, 1)
                    base = torch.where(valid, base, last_base)
            return self._activate_nonnegative(base + self.decoder_scale * od_delta)
        if self.decoder_mode == "residual_mean":
            base = x.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1, -1)
            return self._activate_nonnegative(base + self.decoder_scale * od_delta)
        raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")

    def _decode_full_rank_pairs(
        self,
        x: torch.Tensor,
        future_context: torch.Tensor,
        core_pred: torch.Tensor,
        poi_features: torch.Tensor | None,
        prev_y: torch.Tensor | None,
        prev_valid: torch.Tensor | None,
        local_prediction: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # x [B,L,N,N], seasonal [B,H,N,N] -> local [B,N,N,K].
        last_base = x[:, -1:].expand(-1, self.pred_len, -1, -1)
        seasonal = prev_y.to(device=x.device, dtype=x.dtype) if prev_y is not None else last_base
        if prev_valid is not None:
            valid = prev_valid.to(device=x.device, dtype=torch.bool).view(-1, 1, 1, 1)
            seasonal = torch.where(valid, seasonal, last_base)
        local_input = torch.cat(
            [
                self._transform_pair_history(x).permute(0, 2, 3, 1),
                self._transform_pair_history(seasonal).permute(0, 2, 3, 1),
            ],
            dim=-1,
        )
        local_feature = self.pair_local_encoder(local_input)

        # Every OD pair receives its horizon-specific LLM context before nonlinear fusion:
        # local/context/node features -> pair_feature [B,H,N,N,K].
        context_feature = self.pair_context_proj(future_context).unsqueeze(2).unsqueeze(3)
        origin_feature = self.pair_origin_embedding.weight.view(1, 1, self.num_nodes, 1, -1)
        dest_feature = self.pair_dest_embedding.weight.view(1, 1, 1, self.num_nodes, -1)
        pair_feature = (
            local_feature.unsqueeze(1)
            + context_feature
            + origin_feature
            + dest_feature
        )
        if self.origin_marginal_encoder is not None:
            origin_history = self._transform_pair_history(x.sum(dim=-1)).permute(0, 2, 1)
            dest_history = self._transform_pair_history(x.sum(dim=-2)).permute(0, 2, 1)
            origin_seasonal = self._transform_pair_history(seasonal.sum(dim=-1)).unsqueeze(-1)
            dest_seasonal = self._transform_pair_history(seasonal.sum(dim=-2)).unsqueeze(-1)
            origin_input = torch.cat(
                [
                    origin_history.unsqueeze(1).expand(-1, self.pred_len, -1, -1),
                    origin_seasonal,
                ],
                dim=-1,
            )
            dest_input = torch.cat(
                [
                    dest_history.unsqueeze(1).expand(-1, self.pred_len, -1, -1),
                    dest_seasonal,
                ],
                dim=-1,
            )
            pair_feature = (
                pair_feature
                + self.origin_marginal_encoder(origin_input).unsqueeze(3)
                + self.dest_marginal_encoder(dest_input).unsqueeze(2)
            )

        if self.pair_poi_norm is not None:
            poi = poi_features if poi_features is not None else self.tokenizer.poi_features
            if poi is None:
                raise ValueError("Full-rank POI fusion is enabled, but no POI features were provided.")
            poi = self.pair_poi_norm(poi.to(device=x.device, dtype=torch.float32))
            poi_origin = self.pair_poi_origin_proj(poi)
            poi_dest = self.pair_poi_dest_proj(poi)
            if poi.dim() == 2:
                pair_feature = pair_feature + poi_origin.view(1, 1, self.num_nodes, 1, -1)
                pair_feature = pair_feature + poi_dest.view(1, 1, 1, self.num_nodes, -1)
            else:
                pair_feature = pair_feature + poi_origin.unsqueeze(1).unsqueeze(3)
                pair_feature = pair_feature + poi_dest.unsqueeze(1).unsqueeze(2)

        fused = self.full_rank_pair_fusion(pair_feature)
        base_gate = torch.sigmoid(self.full_rank_base_gate_head(fused).squeeze(-1))
        pair_delta = self.full_rank_delta_head(fused).squeeze(-1)
        if self.full_rank_delta_limit > 0:
            pair_delta = self.full_rank_delta_limit * torch.tanh(pair_delta / self.full_rank_delta_limit)
        relative_delta = None
        if self.full_rank_relative_head is not None:
            relative_delta = torch.tanh(self.full_rank_relative_head(fused).squeeze(-1))
            relative_delta = self.relative_pair_delta_limit * relative_delta

        # base_gate/pair_delta/low_rank_delta: [B,H,N,N].
        base = base_gate * seasonal + (1.0 - base_gate) * last_base
        if prev_valid is not None:
            valid = prev_valid.to(device=x.device, dtype=torch.bool).view(-1, 1, 1, 1)
            base = torch.where(valid, base, last_base)
        low_rank_delta = self.tokenizer.reconstruct(core_pred, poi_features=poi_features)
        if self.delta_limit > 0:
            low_rank_delta = self.delta_limit * torch.tanh(low_rank_delta / self.delta_limit)
        positive_magnitude = self._activate_nonnegative(
            base
            + (base * relative_delta if relative_delta is not None else 0.0)
            + self.decoder_scale * low_rank_delta
            + self.full_rank_delta_scale * pair_delta
        )
        occurrence_logits = None
        occurrence_probability = None
        prediction = positive_magnitude
        if self.full_rank_occurrence_head is not None:
            history_presence = (x > 0).float().mean(dim=1, keepdim=True)
            seasonal_presence = (seasonal > 0).float()
            occurrence_prior = (0.5 * history_presence + 0.5 * seasonal_presence).clamp(1e-4, 1.0 - 1e-4)
            prior_logit = torch.logit(occurrence_prior)
            occurrence_logits = (
                self.full_rank_occurrence_head(fused).squeeze(-1)
                + self.occurrence_prior_scale * prior_logit
            )
            calibration_bias = self._horizon_parameter(
                self.occurrence_calibration_bias,
                occurrence_logits.device,
                occurrence_logits.dtype,
            )
            calibration_temperature = self._horizon_parameter(
                self.occurrence_calibration_temperature,
                occurrence_logits.device,
                occurrence_logits.dtype,
            ).clamp_min(1e-4)
            occurrence_probability = torch.sigmoid(
                (occurrence_logits + calibration_bias)
                / (self.occurrence_temperature * calibration_temperature)
            )
            probability_threshold = self._horizon_parameter(
                self.occurrence_probability_threshold,
                occurrence_probability.device,
                occurrence_probability.dtype,
            )
            if torch.any(probability_threshold > 0):
                occurrence_probability = torch.where(
                    occurrence_probability >= probability_threshold,
                    occurrence_probability,
                    torch.zeros_like(occurrence_probability),
                )
            prediction = occurrence_probability * positive_magnitude
            value_threshold = self._horizon_parameter(
                self.prediction_value_threshold,
                prediction.device,
                prediction.dtype,
            )
            if torch.any(value_threshold > 0):
                prediction = torch.where(
                    prediction >= value_threshold,
                    prediction,
                    torch.zeros_like(prediction),
                )
            calibration_power = self._horizon_parameter(
                self.prediction_calibration_power,
                prediction.device,
                prediction.dtype,
            ).clamp_min(1e-4)
            output_scale = self._horizon_parameter(
                self.output_calibration_scale,
                prediction.device,
                prediction.dtype,
            )
            prediction = output_scale * prediction.clamp_min(0.0).pow(calibration_power)
            if self.evidence_calibration_weights is not None:
                weights = torch.as_tensor(
                    self.evidence_calibration_weights,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                if weights.shape != (self.pred_len, 8):
                    raise ValueError(
                        f"evidence_calibration_weights must have shape [{self.pred_len},8], "
                        f"got {tuple(weights.shape)}"
                    )
                history_mean = x.mean(dim=1, keepdim=True).expand_as(prediction)
                history_max = x.max(dim=1, keepdim=True).values.expand_as(prediction)
                evidence = torch.stack(
                    [
                        torch.ones_like(prediction),
                        prediction,
                        positive_magnitude,
                        seasonal,
                        last_base,
                        history_mean,
                        history_max,
                        occurrence_probability,
                    ],
                    dim=-1,
                )
                prediction = torch.einsum("bhijf,hf->bhij", evidence, weights).clamp_min(0.0)
                evidence_threshold = self._horizon_parameter(
                    self.evidence_calibration_threshold,
                    prediction.device,
                    prediction.dtype,
                )
                if torch.any(evidence_threshold > 0):
                    prediction = torch.where(
                        prediction >= evidence_threshold,
                        prediction,
                        torch.zeros_like(prediction),
                    )
        local_global_gate = None
        if local_prediction is not None:
            if self.local_global_gate_head is None:
                raise RuntimeError("Local OD mixer prediction was provided without a local-global gate.")
            local_global_gate = torch.tanh(self.local_global_gate_head(fused).squeeze(-1))
            local_prediction = local_prediction.to(device=prediction.device, dtype=prediction.dtype)
            prediction = (
                local_prediction
                + local_global_gate * (prediction - local_prediction)
            ).clamp_min(0.0)
        return (
            prediction,
            base_gate,
            pair_delta,
            relative_delta,
            occurrence_logits,
            positive_magnitude,
            local_global_gate,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_time: torch.Tensor | None = None,
        y_time: torch.Tensor | None = None,
        poi_features: torch.Tensor | None = None,
        prev_x: torch.Tensor | None = None,
        prev_y: torch.Tensor | None = None,
        prev_valid: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        local_prediction = None
        if self.local_od_mixer is not None:
            if all(not parameter.requires_grad for parameter in self.local_od_mixer.parameters()):
                with torch.no_grad():
                    local_prediction = self.local_od_mixer(x, prev_x=prev_x)
            else:
                local_prediction = self.local_od_mixer(x, prev_x=prev_x)

        tokens, history_core = self.tokenizer(x, poi_features=poi_features)  # [B, L*r*r, d_model]
        llm_tokens = self._adapt_tokens(tokens)  # [B, L*r*r, llm_dim]
        if self.use_time_features:
            if x_time is None or y_time is None:
                raise ValueError("ODLLM use_time_features=true requires x_time and y_time.")
            # x_time: [B,L,F] -> [B,L*r*r,D], aligned with OD latent tokens.
            history_time = self.history_time_proj(x_time.to(device=x.device, dtype=torch.float32))
            history_time = history_time.unsqueeze(2).expand(-1, -1, self.rank * self.rank, -1)
            history_time = history_time.reshape(x.shape[0], self.input_len * self.rank * self.rank, self.llm_dim)
            llm_tokens = llm_tokens + history_time

        prompt = self.virtual_prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
        llm_input = torch.cat([prompt, llm_tokens], dim=1)
        # llm_input: [B, prompt+L*r*r, llm_dim]. Align dtype with LLM weights.
        llm_input = llm_input.to(self.llm_dtype)
        hidden = self._last_hidden_state(self.llm_model(inputs_embeds=llm_input)).float()
        hidden_tokens = hidden[:, self.num_virtual_prompt_tokens :, :]

        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.context_pooling == "mean":
            context = self.context_norm(hidden_tokens.mean(dim=1))  # [B, llm_dim]
            future_context = context.unsqueeze(1) + horizon
        elif self.context_pooling == "horizon_attention":
            # Horizon-specific queries attend to history OD tokens:
            # query [B,H,D], key/value [B,L*r*r,D] -> future_context [B,H,D].
            attended, _ = self.horizon_attention(horizon, hidden_tokens, hidden_tokens, need_weights=False)
            future_context = self.context_norm(horizon + attended)
        else:
            raise ValueError(f"Unsupported context_pooling: {self.context_pooling}")
        if self.use_time_features:
            future_time = self.future_time_proj(y_time.to(device=x.device, dtype=torch.float32))
            future_context = future_context + future_time

        core_flat = self.core_head(future_context)
        core_pred = core_flat.reshape(x.shape[0], self.pred_len, self.rank, self.rank)
        base_blend = None
        if self.context_base_gate_head is not None and prev_y is not None:
            base_blend = torch.sigmoid(self.context_base_gate_head(future_context)).view(
                x.shape[0], self.pred_len, 1, 1
            )
        pair_gate = None
        if self.decoder_pair_gate_head is not None:
            pair_gate = torch.sigmoid(self.decoder_pair_gate_head(future_context)).view(
                x.shape[0], self.pred_len, 1, 1
            )
        full_rank_base_gate = None
        full_rank_pair_delta = None
        full_rank_relative_delta = None
        occurrence_logits = None
        positive_magnitude = None
        local_global_gate = None
        if self.use_full_rank_pair_decoder:
            (
                y_pred,
                full_rank_base_gate,
                full_rank_pair_delta,
                full_rank_relative_delta,
                occurrence_logits,
                positive_magnitude,
                local_global_gate,
            ) = self._decode_full_rank_pairs(
                    x,
                    future_context,
                    core_pred,
                    poi_features,
                    prev_y,
                    prev_valid,
                    local_prediction,
                )
        else:
            y_pred = self._decode(
                x,
                core_pred,
                poi_features=poi_features,
                prev_y=prev_y,
                prev_valid=prev_valid,
                base_blend=base_blend,
                pair_gate=pair_gate,
            )

        if return_latent or occurrence_logits is not None:
            output = {
                "prediction": y_pred,
                "occurrence_logits": occurrence_logits,
                "positive_magnitude": positive_magnitude,
            }
            if return_latent:
                output.update(
                    {
                        "history_core": history_core,
                        "future_core": core_pred,
                        "tokens": tokens,
                        "llm_tokens": llm_tokens,
                        "hidden_tokens": hidden_tokens,
                        "seasonal_blend": torch.sigmoid(self.seasonal_blend_logit).detach(),
                        "context_base_blend": None if base_blend is None else base_blend.detach(),
                        "decoder_pair_gate": None if pair_gate is None else pair_gate.detach(),
                        "full_rank_base_gate": (
                            None if full_rank_base_gate is None else full_rank_base_gate.detach()
                        ),
                        "full_rank_pair_delta": (
                            None if full_rank_pair_delta is None else full_rank_pair_delta.detach()
                        ),
                        "full_rank_relative_delta": (
                            None if full_rank_relative_delta is None else full_rank_relative_delta.detach()
                        ),
                        "local_prediction": (
                            None if local_prediction is None else local_prediction.detach()
                        ),
                        "local_global_gate": (
                            None if local_global_gate is None else local_global_gate.detach()
                        ),
                    }
                )
            return output
        return y_pred

    @staticmethod
    def _last_hidden_state(outputs) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, tuple):
            return outputs[0]
        raise RuntimeError("Cannot find hidden states in LLM output.")
