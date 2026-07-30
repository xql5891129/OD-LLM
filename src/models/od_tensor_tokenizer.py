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

    def __init__(
        self,
        num_nodes: int,
        rank: int,
        d_model: int,
        max_input_len: int,
        dropout: float = 0.1,
        poi_features: torch.Tensor | None = None,
        poi_feature_dim: int = 0,
        use_poi_features: bool = False,
        poi_projection_scale: float = 0.1,
        value_transform: str = "none",
        projection_mode: str = "learnable",
        projection_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.d_model = d_model
        self.max_input_len = max_input_len
        self.use_poi_features = use_poi_features and poi_feature_dim > 0
        self.poi_projection_scale = poi_projection_scale
        self.value_transform = value_transform
        self.projection_mode = str(projection_mode).lower()
        if self.projection_mode not in {"learnable", "svd_fixed", "svd_trainable"}:
            raise ValueError(
                "projection_mode must be one of learnable, svd_fixed, or svd_trainable; "
                f"got {projection_mode!r}"
            )
        if self.projection_mode == "learnable" and projection_init is not None:
            raise ValueError("projection_init is only valid for SVD projection modes.")
        if self.projection_mode != "learnable" and projection_init is None:
            raise ValueError(f"projection_mode={self.projection_mode} requires precomputed Po/Pd.")

        self.Po = nn.Parameter(torch.empty(num_nodes, rank))
        self.Pd = nn.Parameter(torch.empty(num_nodes, rank))
        if projection_init is None:
            nn.init.xavier_uniform_(self.Po)
            nn.init.xavier_uniform_(self.Pd)
        else:
            po, pd = projection_init
            expected_shape = (num_nodes, rank)
            if tuple(po.shape) != expected_shape or tuple(pd.shape) != expected_shape:
                raise ValueError(
                    "Precomputed projection shapes must both equal "
                    f"{expected_shape}; got Po={tuple(po.shape)}, Pd={tuple(pd.shape)}"
                )
            with torch.no_grad():
                self.Po.copy_(po.detach().to(dtype=self.Po.dtype))
                self.Pd.copy_(pd.detach().to(dtype=self.Pd.dtype))
        if self.projection_mode == "svd_fixed":
            self.Po.requires_grad_(False)
            self.Pd.requires_grad_(False)

        self.value_proj = nn.Linear(1, d_model)
        self.basis_embedding = nn.Embedding(rank * rank, d_model)
        self.time_embedding = nn.Embedding(max_input_len, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if self.use_poi_features:
            if poi_features is not None and tuple(poi_features.shape) != (num_nodes, poi_feature_dim):
                raise ValueError(
                    f"Expected poi_features shape [{num_nodes},{poi_feature_dim}], got {tuple(poi_features.shape)}"
                )
            if poi_features is not None:
                self.register_buffer("poi_features", poi_features.float(), persistent=False)
            else:
                self.poi_features = None
            self.poi_norm = nn.LayerNorm(poi_feature_dim)
            self.poi_origin_proj = nn.Linear(poi_feature_dim, rank)
            self.poi_dest_proj = nn.Linear(poi_feature_dim, rank)
        else:
            self.poi_features = None
            self.poi_norm = None
            self.poi_origin_proj = None
            self.poi_dest_proj = None

    def _transform_core_values(self, core: torch.Tensor) -> torch.Tensor:
        if self.value_transform == "none":
            return core
        if self.value_transform in {"signed_log1p", "symlog"}:
            return core.sign() * torch.log1p(core.abs())
        if self.value_transform == "log1p":
            return torch.log1p(core.clamp_min(0.0))
        raise ValueError(f"Unsupported tokenizer value_transform: {self.value_transform}")

    def _effective_projections(self, poi_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return origin/destination basis matrices, optionally POI-aware.

        Static mode returns Po/Pd with shape [N, r]. Dynamic line-level mode
        accepts sample POI features [B, N, F] and returns [B, N, r].
        """
        if not self.use_poi_features:
            return self.Po, self.Pd
        poi = poi_features if poi_features is not None else self.poi_features
        if poi is None:
            raise ValueError("POI-guided tokenizer is enabled, but no POI features were provided.")
        poi = poi.to(device=self.Po.device, dtype=self.Po.dtype)
        poi = self.poi_norm(poi)
        po = self.Po + self.poi_projection_scale * self.poi_origin_proj(poi)
        pd = self.Pd + self.poi_projection_scale * self.poi_dest_proj(poi)
        return po, pd

    def forward(
        self,
        x: torch.Tensor,
        poi_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        po, pd = self._effective_projections(poi_features=poi_features)
        if po.dim() == 2:
            # core[b, l, a, c] = sum_i sum_j Po[i, a] * x[b, l, i, j] * Pd[j, c]
            core = torch.einsum("ia,blij,jc->blac", po, x, pd)
        else:
            # Dynamic line-level POI: Po/Pd are sample-specific, [B, N, r].
            core = torch.einsum("bia,blij,bjc->blac", po, x, pd)
        flat_core = self._transform_core_values(core).reshape(bsz, input_len, self.rank * self.rank)

        value_tokens = self.value_proj(flat_core.unsqueeze(-1))
        basis_ids = torch.arange(self.rank * self.rank, device=x.device)
        basis_tokens = self.basis_embedding(basis_ids).view(1, 1, self.rank * self.rank, self.d_model)
        time_ids = torch.arange(input_len, device=x.device)
        time_tokens = self.time_embedding(time_ids).view(1, input_len, 1, self.d_model)

        tokens = value_tokens + basis_tokens + time_tokens
        tokens = tokens.reshape(bsz, input_len * self.rank * self.rank, self.d_model)
        return self.dropout(self.token_norm(tokens)), core

    def reconstruct(self, core: torch.Tensor, poi_features: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct OD matrices from latent core.

        Args:
            core: Future latent core, shape [B, H, r, r].

        Returns:
            od: Reconstructed OD matrices, shape [B, H, N, N].
        """
        if core.dim() != 4:
            raise ValueError(f"Expected core shape [B, H, r, r], got {tuple(core.shape)}")
        po, pd = self._effective_projections(poi_features=poi_features)
        if po.dim() == 2:
            return torch.einsum("ia,bhac,jc->bhij", po, core, pd)
        return torch.einsum("bia,bhac,bjc->bhij", po, core, pd)

    def basis_heatmap(self, origin_dim: int, dest_dim: int) -> torch.Tensor:
        """Return OD basis heatmap outer(Po[:, a], Pd[:, b]), shape [N, N]."""
        po, pd = self._effective_projections()
        return torch.outer(po[:, origin_dim], pd[:, dest_dim])
