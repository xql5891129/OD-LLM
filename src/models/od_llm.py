from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_blocks import MiniGPTBackbone, ReprogrammingLayer
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
        freeze_llm: bool = True,
        use_reprogramming: bool = False,
        num_virtual_prompt_tokens: int = 8,
        num_source_tokens: int = 1000,
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
        )
        self.llm_model, real_llm_dim, source_embeddings = self._build_backbone(
            llm_model=llm_model,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
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

        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        if use_reprogramming and source_embeddings is not None:
            d_keys = max(8, d_model // max(llm_heads, 1))
            self.input_adapter = ReprogrammingLayer(d_model, llm_heads, d_keys, self.llm_dim, dropout)
            self.register_buffer("source_embeddings", source_embeddings, persistent=False)
        else:
            self.input_adapter = nn.Linear(d_model, self.llm_dim)
            self.source_embeddings = None

        self.virtual_prompt = nn.Parameter(torch.randn(num_virtual_prompt_tokens, self.llm_dim) * 0.02)
        self.context_norm = nn.LayerNorm(self.llm_dim)
        self.horizon_embedding = nn.Embedding(pred_len, self.llm_dim)
        self.core_head = nn.Sequential(
            nn.Linear(self.llm_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, rank * rank),
        )

    def _build_backbone(
        self,
        llm_model: str,
        pretrained: bool,
        pretrained_path: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: str | None,
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

    @staticmethod
    def _infer_hidden_size(config, model: nn.Module) -> int:
        for attr in ["hidden_size", "n_embd", "d_model"]:
            value = getattr(config, attr, None)
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

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        tokens, history_core = self.tokenizer(x)  # [B, L*r*r, d_model]
        llm_tokens = self._adapt_tokens(tokens)  # [B, L*r*r, llm_dim]

        prompt = self.virtual_prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
        llm_input = torch.cat([prompt, llm_tokens], dim=1)
        # llm_input: [B, prompt+L*r*r, llm_dim]. Align dtype with LLM weights.
        llm_input = llm_input.to(self.llm_dtype)
        hidden = self._last_hidden_state(self.llm_model(inputs_embeds=llm_input)).float()
        hidden_tokens = hidden[:, self.num_virtual_prompt_tokens :, :]

        context = self.context_norm(hidden_tokens.mean(dim=1))  # [B, llm_dim]
        horizon_ids = torch.arange(self.pred_len, device=x.device)
        future_context = context.unsqueeze(1) + self.horizon_embedding(horizon_ids).unsqueeze(0)

        core_flat = self.core_head(future_context)
        core_pred = core_flat.reshape(x.shape[0], self.pred_len, self.rank, self.rank)
        y_pred = F.softplus(self.tokenizer.reconstruct(core_pred))

        if return_latent:
            return {
                "prediction": y_pred,
                "history_core": history_core,
                "future_core": core_pred,
                "tokens": tokens,
                "llm_tokens": llm_tokens,
                "hidden_tokens": hidden_tokens,
            }
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
