from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return (pred[mask] - true[mask]).abs().mean()
    return pred.new_tensor(0.0)


def _masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return (pred[mask] - true[mask]).square().mean()
    return pred.new_tensor(0.0)


def sparse_od_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    topk: int = 20,
    nonzero_threshold: float = 0.0,
    mse_weight: float = 0.0,
    nonzero_mse_weight: float = 0.0,
    topk_mse_weight: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    occurrence_logits: torch.Tensor | None = None,
    positive_magnitude: torch.Tensor | None = None,
    occurrence_loss_weight: float = 0.0,
    occurrence_positive_weight: float = 0.7,
    magnitude_loss_weight: float = 0.0,
    magnitude_mse_weight: float = 0.0,
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

    # Loss terms, especially MSE on sparse high-flow OD pairs, are intentionally
    # accumulated in fp32. Under fp16 autocast, squaring a rare large prediction
    # can overflow before GradScaler has a chance to skip the update.
    pred = pred.float()
    true = true.float()

    if valid_mask is None:
        valid_mask = torch.ones_like(true, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=true.device, dtype=torch.bool)
        if valid_mask.ndim == true.ndim - 1 and valid_mask.shape[0] == true.shape[0]:
            valid_mask = valid_mask.unsqueeze(1)
        elif valid_mask.ndim == 2 and true.ndim == 4:
            valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
        if valid_mask.shape != true.shape:
            valid_mask = valid_mask.expand_as(true)

    if valid_mask.any():
        diff = pred - true
        loss_all = diff.abs()[valid_mask].mean()
        loss_mse = diff.square()[valid_mask].mean()
    else:
        loss_all = pred.new_tensor(0.0)
        loss_mse = pred.new_tensor(0.0)

    nonzero_mask = (true > nonzero_threshold) & valid_mask
    loss_nonzero = _masked_mae(pred, true, nonzero_mask)
    loss_nonzero_mse = _masked_mse(pred, true, nonzero_mask)

    flat_pred = pred.reshape(-1, pred.shape[-1] * pred.shape[-2])
    flat_true = true.reshape_as(flat_pred)
    flat_valid = valid_mask.reshape_as(flat_true)
    k = min(int(topk), flat_true.shape[-1])
    if k > 0 and flat_valid.any():
        masked_true = flat_true.masked_fill(~flat_valid, float("-inf"))
        topk_idx = torch.topk(masked_true, k=k, dim=-1).indices
        topk_pred = torch.gather(flat_pred, dim=-1, index=topk_idx)
        topk_true = torch.gather(flat_true, dim=-1, index=topk_idx)
        topk_valid = torch.gather(flat_valid, dim=-1, index=topk_idx)
        if topk_valid.any():
            topk_diff = topk_pred - topk_true
            loss_topk = topk_diff.abs()[topk_valid].mean()
            loss_topk_mse = topk_diff.square()[topk_valid].mean()
        else:
            loss_topk = pred.new_tensor(0.0)
            loss_topk_mse = pred.new_tensor(0.0)
    else:
        loss_topk = pred.new_tensor(0.0)
        loss_topk_mse = pred.new_tensor(0.0)

    loss = (
        alpha * loss_all
        + beta * loss_nonzero
        + gamma * loss_topk
        + mse_weight * loss_mse
        + nonzero_mse_weight * loss_nonzero_mse
        + topk_mse_weight * loss_topk_mse
    )
    loss_occurrence = pred.new_tensor(0.0)
    if occurrence_logits is not None and occurrence_loss_weight > 0.0:
        logits = occurrence_logits.float()
        target_presence = (true > nonzero_threshold).float()
        positive_mask = (target_presence > 0.5) & valid_mask
        negative_mask = (target_presence <= 0.5) & valid_mask
        positive_loss = (
            F.binary_cross_entropy_with_logits(logits[positive_mask], target_presence[positive_mask])
            if positive_mask.any()
            else pred.new_tensor(0.0)
        )
        negative_loss = (
            F.binary_cross_entropy_with_logits(logits[negative_mask], target_presence[negative_mask])
            if negative_mask.any()
            else pred.new_tensor(0.0)
        )
        positive_weight = min(max(float(occurrence_positive_weight), 0.0), 1.0)
        loss_occurrence = positive_weight * positive_loss + (1.0 - positive_weight) * negative_loss
        loss = loss + occurrence_loss_weight * loss_occurrence

    loss_magnitude = pred.new_tensor(0.0)
    loss_magnitude_mse = pred.new_tensor(0.0)
    if positive_magnitude is not None and (magnitude_loss_weight > 0.0 or magnitude_mse_weight > 0.0):
        magnitude = positive_magnitude.float()
        loss_magnitude = _masked_mae(magnitude, true, nonzero_mask)
        loss_magnitude_mse = _masked_mse(magnitude, true, nonzero_mask)
        loss = loss + magnitude_loss_weight * loss_magnitude + magnitude_mse_weight * loss_magnitude_mse
    parts = {
        "loss_all": float(loss_all.detach().cpu().item()),
        "loss_nonzero": float(loss_nonzero.detach().cpu().item()),
        "loss_topk": float(loss_topk.detach().cpu().item()),
        "loss_mse": float(loss_mse.detach().cpu().item()),
        "loss_nonzero_mse": float(loss_nonzero_mse.detach().cpu().item()),
        "loss_topk_mse": float(loss_topk_mse.detach().cpu().item()),
        "loss_occurrence": float(loss_occurrence.detach().cpu().item()),
        "loss_magnitude": float(loss_magnitude.detach().cpu().item()),
        "loss_magnitude_mse": float(loss_magnitude_mse.detach().cpu().item()),
    }
    return loss, parts
