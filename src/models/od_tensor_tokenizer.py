from __future__ import annotations

import torch
import torch.nn as nn


class ODTensorTokenizer(nn.Module):
    """Learnable two-sided projection tokenizer for OD matrices.

    Given x with shape [B, L, N, N], learn Po and Pd with shape [N, r].
    For each time step:

        core_t = Po^T x_t Pd

    The core tensor has shape [B, L, r, r]. Each scalar core entry becomes one
    OD latent token, so the final token sequence has shape [B, L*r*r, d_model].
    """

    def __init__(self, num_nodes: int, rank: int, d_model: int, max_input_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.d_model = d_model
        self.max_input_len = max_input_len

        self.Po = nn.Parameter(torch.empty(num_nodes, rank))
        self.Pd = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.xavier_uniform_(self.Po)
        nn.init.xavier_uniform_(self.Pd)

        self.value_proj = nn.Linear(1, d_model)
        self.basis_embedding = nn.Embedding(rank * rank, d_model)
        self.time_embedding = nn.Embedding(max_input_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize OD history.

        Args:
            x: OD history, shape [B, L, N, N].

        Returns:
            tokens: OD latent tokens, shape [B, L*r*r, d_model].
            core: Compressed OD core, shape [B, L, r, r].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B, L, N, N], got {tuple(x.shape)}")
        bsz, input_len, n_origin, n_dest = x.shape
        if n_origin != self.num_nodes or n_dest != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got {n_origin}x{n_dest}")
        if input_len > self.max_input_len:
            raise ValueError(f"input_len={input_len} exceeds max_input_len={self.max_input_len}")

        # core[b, l, a, c] = sum_i sum_j Po[i, a] * x[b, l, i, j] * Pd[j, c]
        core = torch.einsum("ia,blij,jc->blac", self.Po, x, self.Pd)
        flat_core = core.reshape(bsz, input_len, self.rank * self.rank)

        value_tokens = self.value_proj(flat_core.unsqueeze(-1))
        basis_ids = torch.arange(self.rank * self.rank, device=x.device)
        basis_tokens = self.basis_embedding(basis_ids).view(1, 1, self.rank * self.rank, self.d_model)
        time_ids = torch.arange(input_len, device=x.device)
        time_tokens = self.time_embedding(time_ids).view(1, input_len, 1, self.d_model)

        tokens = value_tokens + basis_tokens + time_tokens
        tokens = tokens.reshape(bsz, input_len * self.rank * self.rank, self.d_model)
        return self.dropout(tokens), core

    def reconstruct(self, core: torch.Tensor) -> torch.Tensor:
        """Reconstruct OD matrices from latent core.

        Args:
            core: Future latent core, shape [B, H, r, r].

        Returns:
            od: Reconstructed OD matrices, shape [B, H, N, N].
        """
        if core.dim() != 4:
            raise ValueError(f"Expected core shape [B, H, r, r], got {tuple(core.shape)}")
        return torch.einsum("ia,bhac,jc->bhij", self.Po, core, self.Pd)

    def basis_heatmap(self, origin_dim: int, dest_dim: int) -> torch.Tensor:
        """Return OD basis heatmap outer(Po[:, a], Pd[:, b]), shape [N, N]."""
        return torch.outer(self.Po[:, origin_dim], self.Pd[:, dest_dim])

