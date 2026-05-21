from __future__ import annotations

import torch


def _masked_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return (pred[mask] - true[mask]).abs().mean()
    return pred.new_tensor(0.0)


def sparse_od_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    topk: int = 20,
    nonzero_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sparse OD loss.

    Args:
        pred: Predicted OD flow, shape [B, H, N, N].
        true: Ground-truth OD flow, shape [B, H, N, N].

    Returns:
        loss: Weighted scalar loss.
        parts: Detached scalar loss parts for logging.
    """
    if pred.shape != true.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != true shape {tuple(true.shape)}")

    loss_all = (pred - true).abs().mean()
    nonzero_mask = true > nonzero_threshold
    loss_nonzero = _masked_mae(pred, true, nonzero_mask)

    flat_pred = pred.reshape(-1, pred.shape[-1] * pred.shape[-2])
    flat_true = true.reshape_as(flat_pred)
    k = min(int(topk), flat_true.shape[-1])
    if k > 0:
        topk_idx = torch.topk(flat_true, k=k, dim=-1).indices
        topk_pred = torch.gather(flat_pred, dim=-1, index=topk_idx)
        topk_true = torch.gather(flat_true, dim=-1, index=topk_idx)
        loss_topk = (topk_pred - topk_true).abs().mean()
    else:
        loss_topk = pred.new_tensor(0.0)

    loss = alpha * loss_all + beta * loss_nonzero + gamma * loss_topk
    parts = {
        "loss_all": float(loss_all.detach().cpu().item()),
        "loss_nonzero": float(loss_nonzero.detach().cpu().item()),
        "loss_topk": float(loss_topk.detach().cpu().item()),
    }
    return loss, parts

