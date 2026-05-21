from __future__ import annotations

import math
from typing import Any

import torch


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    topk: int = 20,
    high_flow_quantile: float = 0.9,
    zero_threshold: float = 0.0,
    pred_positive_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute OD metrics.

    Args:
        pred: Predicted OD flow, shape [B, H, N, N].
        true: Ground-truth OD flow, shape [B, H, N, N].
    """
    pred = pred.detach().float().cpu()
    true = true.detach().float().cpu()
    abs_err = (pred - true).abs()
    sq_err = (pred - true).pow(2)

    mae = abs_err.mean()
    rmse = torch.sqrt(sq_err.mean())
    wape = abs_err.sum() / true.abs().sum().clamp_min(1e-8)

    nonzero_mask = true > zero_threshold
    nonzero_mae = _safe_mean(abs_err[nonzero_mask])

    flat_true = true.reshape(-1, true.shape[-1] * true.shape[-2])
    flat_pred = pred.reshape_as(flat_true)
    k = min(topk, flat_true.shape[-1])
    if k > 0:
        topk_idx = torch.topk(flat_true, k=k, dim=-1).indices
        topk_true = torch.gather(flat_true, dim=-1, index=topk_idx)
        topk_pred = torch.gather(flat_pred, dim=-1, index=topk_idx)
        topk_mae = (topk_pred - topk_true).abs().mean()
    else:
        topk_mae = pred.new_tensor(0.0)

    positive_true = true[true > zero_threshold]
    if positive_true.numel() > 0:
        threshold = torch.quantile(positive_true, high_flow_quantile)
        high_mask = true >= threshold
        high_flow_mae = _safe_mean(abs_err[high_mask])
    else:
        high_flow_mae = pred.new_tensor(0.0)

    zero_mask = true <= zero_threshold
    if zero_mask.any():
        zero_false_positive_rate = ((pred > pred_positive_threshold) & zero_mask).float().sum() / zero_mask.float().sum()
    else:
        zero_false_positive_rate = pred.new_tensor(0.0)

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "wape": float(wape.item()),
        "nonzero_mae": float(nonzero_mae.item()),
        "topk_mae": float(topk_mae.item()),
        "high_flow_mae": float(high_flow_mae.item()),
        "zero_false_positive_rate": float(zero_false_positive_rate.item()),
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float) and math.isfinite(value):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
