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
    y_time: torch.Tensor | None = None,
    topk: int = 20,
    high_flow_quantile: float = 0.9,
    high_flow_max_samples: int = 1_000_000,
    zero_threshold: float = 0.0,
    pred_positive_threshold: float = 0.5,
    peak_hours: tuple[int, ...] = (7, 8, 9, 17, 18, 19),
) -> dict[str, float]:
    """Compute OD metrics.

    Args:
        pred: Predicted OD flow, shape [B, H, N, N].
        true: Ground-truth OD flow, shape [B, H, N, N].
    """
    pred = pred.detach().float().cpu()
    true = true.detach().float().cpu()
    if y_time is not None:
        y_time = y_time.detach().float().cpu()
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
        threshold_source = positive_true
        if high_flow_max_samples > 0 and positive_true.numel() > high_flow_max_samples:
            # `torch.quantile` has tensor-size limits on large MetroFlow arrays.
            # Use a deterministic uniform sample for the threshold, then score
            # high-flow MAE on the full tensor.
            sample_idx = torch.linspace(
                0,
                positive_true.numel() - 1,
                steps=int(high_flow_max_samples),
                dtype=torch.long,
            )
            threshold_source = positive_true[sample_idx]
        q = min(max(float(high_flow_quantile), 0.0), 1.0)
        kth = max(1, min(threshold_source.numel(), int(math.ceil(q * threshold_source.numel()))))
        threshold = torch.kthvalue(threshold_source, kth).values
        high_mask = true >= threshold
        high_flow_mae = _safe_mean(abs_err[high_mask])
    else:
        high_flow_mae = pred.new_tensor(0.0)

    zero_mask = true <= zero_threshold
    if zero_mask.any():
        zero_false_positive_rate = ((pred > pred_positive_threshold) & zero_mask).float().sum() / zero_mask.float().sum()
    else:
        zero_false_positive_rate = pred.new_tensor(0.0)

    metrics = {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "wape": float(wape.item()),
        "nonzero_mae": float(nonzero_mae.item()),
        "topk_mae": float(topk_mae.item()),
        "high_flow_mae": float(high_flow_mae.item()),
        "zero_false_positive_rate": float(zero_false_positive_rate.item()),
    }

    if y_time is not None and y_time.shape[-1] >= 5:
        day_phase = torch.atan2(y_time[..., 0], y_time[..., 1])
        hour = ((day_phase % (2 * math.pi)) / (2 * math.pi) * 24).floor().long()
        peak_tensor = torch.tensor(peak_hours, dtype=torch.long)
        peak_mask = (hour.unsqueeze(-1) == peak_tensor).any(dim=-1)
        offpeak_mask = ~peak_mask
        weekend_mask = y_time[..., 4] > 0.5
        weekday_mask = ~weekend_mask

        def time_mask_mae(mask: torch.Tensor) -> float:
            if not mask.any():
                return 0.0
            expanded = mask[..., None, None].expand_as(abs_err)
            return float(abs_err[expanded].mean().item())

        metrics.update(
            {
                "peak_hour_mae": time_mask_mae(peak_mask),
                "offpeak_mae": time_mask_mae(offpeak_mask),
                "weekend_mae": time_mask_mae(weekend_mask),
                "weekday_mae": time_mask_mae(weekday_mask),
            }
        )

    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float) and math.isfinite(value):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
