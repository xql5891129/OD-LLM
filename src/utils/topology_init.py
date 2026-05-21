from __future__ import annotations

import torch


def random_orthogonal_init(num_nodes: int, rank: int, scale: float = 1.0) -> torch.Tensor:
    mat = torch.randn(num_nodes, rank)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :rank] * scale

