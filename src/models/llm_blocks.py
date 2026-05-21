from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    last_hidden_state: torch.Tensor


class MiniGPTBackbone(nn.Module):
    """Small GPT-style backbone for offline smoke tests.

    This is not a pretrained LLM. It mimics the `inputs_embeds -> last_hidden_state`
    interface so the OD-LLM path can be validated without downloading HuggingFace
    weights. The real GPT-2 path can be enabled through config once transformers
    and local checkpoints are available.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> BackboneOutput:
        if inputs_embeds.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {inputs_embeds.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        pos_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        x = inputs_embeds + self.position_embedding(pos_ids).unsqueeze(0)
        causal_mask = torch.triu(
            torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device),
            diagonal=1,
        )
        hidden = self.encoder(x, mask=causal_mask)
        return BackboneOutput(last_hidden_state=self.norm(hidden))


class ReprogrammingLayer(nn.Module):
    """Time-LLM style reprogramming layer.

    target_embedding: [B, S, d_model]
    source_embedding: [V, d_llm]
    output: [B, S, d_llm]
    """

    def __init__(self, d_model: int, n_heads: int, d_keys: int, d_llm: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_embedding: torch.Tensor, source_embedding: torch.Tensor) -> torch.Tensor:
        bsz, target_len, _ = target_embedding.shape
        source_len, _ = source_embedding.shape
        heads = self.n_heads

        # Align dtype with projection weights (LLM embeddings may be BFloat16)
        target_dtype = self.query_projection.weight.dtype
        target_embedding = target_embedding.to(target_dtype)
        source_embedding = source_embedding.to(target_dtype)

        query = self.query_projection(target_embedding).view(bsz, target_len, heads, -1)
        key = self.key_projection(source_embedding).view(source_len, heads, -1)
        value = self.value_projection(source_embedding).view(source_len, heads, -1)

        scale = query.shape[-1] ** -0.5
        scores = torch.einsum("blhe,she->bhls", query, key)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, value).reshape(bsz, target_len, -1)
        return self.out_projection(out)

