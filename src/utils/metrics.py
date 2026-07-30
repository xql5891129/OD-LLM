from __future__ import annotations

import math
from typing import Any

import torch


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


def _expand_valid_mask(valid_mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    valid = valid_mask.detach().bool().cpu()
    if valid.ndim == target.ndim - 1 and valid.shape[0] == target.shape[0]:
        valid = valid.unsqueeze(1)
    elif valid.ndim == 2 and target.ndim == 4:
        valid = valid.unsqueeze(0).unsqueeze(0)
    if valid.shape != target.shape:
        valid = valid.expand_as(target)
    return valid


def _filter_report_keys(metrics: dict[str, float], report_keys: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    if not report_keys:
        return metrics
    return {key: metrics[key] for key in report_keys if key in metrics}


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    y_time: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    topk: int = 20,
    high_flow_quantile: float = 0.9,
    high_flow_max_samples: int = 1_000_000,
    zero_threshold: float = 0.0,
    pred_positive_threshold: float = 0.5,
    peak_hours: tuple[int, ...] = (7, 8, 9, 17, 18, 19),
    rainy_feature_index: int | None = None,
    rainy_threshold: float = 0.5,
    report_keys: list[str] | tuple[str, ...] | None = None,
    use_od_mask: bool = False,
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
    valid = _expand_valid_mask(valid_mask, true)
    abs_err = (pred - true).abs()
    sq_err = (pred - true).pow(2)

    mae = _safe_mean(abs_err[valid])
    rmse = torch.sqrt(_safe_mean(sq_err[valid]))
    wape = abs_err[valid].sum() / true.abs()[valid].sum().clamp_min(1e-8)

    nonzero_mask = (true > zero_threshold) & valid
    nonzero_mae = _safe_mean(abs_err[nonzero_mask])

    flat_true = true.reshape(-1, true.shape[-1] * true.shape[-2])
    flat_pred = pred.reshape_as(flat_true)
    flat_valid = valid.reshape_as(flat_true)
    k = min(topk, flat_true.shape[-1])
    if k > 0 and flat_valid.any():
        masked_true = flat_true.masked_fill(~flat_valid, float("-inf"))
        topk_idx = torch.topk(masked_true, k=k, dim=-1).indices
        topk_true = torch.gather(flat_true, dim=-1, index=topk_idx)
        topk_pred = torch.gather(flat_pred, dim=-1, index=topk_idx)
        topk_valid = torch.gather(flat_valid, dim=-1, index=topk_idx)
        topk_mae = _safe_mean((topk_pred - topk_true).abs()[topk_valid])
    else:
        topk_mae = pred.new_tensor(0.0)

    positive_true = true[(true > zero_threshold) & valid]
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
        high_mask = (true >= threshold) & valid
        high_flow_mae = _safe_mean(abs_err[high_mask])
    else:
        high_flow_mae = pred.new_tensor(0.0)

    zero_mask = (true <= zero_threshold) & valid
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
            expanded = mask[..., None, None].expand_as(abs_err) & valid
            if not expanded.any():
                return 0.0
            return float(abs_err[expanded].mean().item())

        metrics.update(
            {
                "peak_hour_mae": time_mask_mae(peak_mask),
                "offpeak_mae": time_mask_mae(offpeak_mask),
                "weekend_mae": time_mask_mae(weekend_mask),
                "weekday_mae": time_mask_mae(weekday_mask),
            }
        )
        if rainy_feature_index is not None:
            feature_index = int(rainy_feature_index)
            feature_dim = int(y_time.shape[-1])
            if feature_index < 0:
                feature_index += feature_dim
            if 0 <= feature_index < feature_dim:
                rainy_mask = y_time[..., feature_index] > float(rainy_threshold)
                metrics.update(
                    {
                        "rainy_mae": time_mask_mae(rainy_mask),
                        "nonrainy_mae": time_mask_mae(~rainy_mask),
                    }
                )

    return _filter_report_keys(metrics, report_keys)


class ODMetricAccumulator:
    """Streaming OD metrics for large full-station OD tensors.

    This keeps only scalar sums on CPU instead of concatenating every
    [B, H, N, N] prediction during validation/testing.
    """

    def __init__(
        self,
        topk: int = 20,
        high_flow_quantile: float = 0.9,
        high_flow_max_samples: int = 1_000_000,
        zero_threshold: float = 0.0,
        pred_positive_threshold: float = 0.5,
        peak_hours: tuple[int, ...] | list[int] = (7, 8, 9, 17, 18, 19),
        rainy_feature_index: int | None = None,
        rainy_threshold: float = 0.5,
        report_keys: list[str] | tuple[str, ...] | None = None,
        use_od_mask: bool = False,
    ) -> None:
        self.topk = int(topk)
        self.high_flow_quantile = float(high_flow_quantile)
        self.high_flow_max_samples = int(high_flow_max_samples)
        self.zero_threshold = float(zero_threshold)
        self.pred_positive_threshold = float(pred_positive_threshold)
        self.peak_hours = tuple(int(hour) for hour in peak_hours)
        self.rainy_feature_index = None if rainy_feature_index is None else int(rainy_feature_index)
        self.rainy_threshold = float(rainy_threshold)
        self.report_keys = list(report_keys) if report_keys else None
        self.use_od_mask = bool(use_od_mask)
        self.reset()

    def reset(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.true_abs_sum = 0.0
        self.count = 0

        self.nonzero_abs_sum = 0.0
        self.nonzero_count = 0
        self.topk_abs_sum = 0.0
        self.topk_count = 0
        self.high_flow_abs_sum = 0.0
        self.high_flow_count = 0
        self.zero_fp_count = 0
        self.zero_count = 0

        self.peak_abs_sum = 0.0
        self.peak_count = 0
        self.offpeak_abs_sum = 0.0
        self.offpeak_count = 0
        self.weekend_abs_sum = 0.0
        self.weekend_count = 0
        self.weekday_abs_sum = 0.0
        self.weekday_count = 0
        self.rainy_abs_sum = 0.0
        self.rainy_count = 0
        self.nonrainy_abs_sum = 0.0
        self.nonrainy_count = 0

    @staticmethod
    def _ratio(total: float, count: int | float) -> float:
        return 0.0 if count <= 0 else float(total / count)

    def _update_time_mask(
        self,
        abs_err: torch.Tensor,
        y_time: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if y_time.shape[-1] < 5:
            return

        # Work at [B, H] granularity, then aggregate only valid OD cells.
        if valid_mask is None:
            per_slice_abs = abs_err.sum(dim=(-1, -2))
            per_slice_count = torch.full_like(per_slice_abs, int(abs_err.shape[-1] * abs_err.shape[-2]))
        else:
            valid = valid_mask.bool()
            per_slice_abs = abs_err.masked_fill(~valid, 0.0).sum(dim=(-1, -2))
            per_slice_count = valid.sum(dim=(-1, -2)).to(per_slice_abs.dtype)

        day_phase = torch.atan2(y_time[..., 0], y_time[..., 1])
        hour = ((day_phase % (2 * math.pi)) / (2 * math.pi) * 24).floor().long()
        peak_tensor = torch.tensor(self.peak_hours, dtype=torch.long)
        peak_mask = (hour.unsqueeze(-1) == peak_tensor).any(dim=-1)
        offpeak_mask = ~peak_mask
        weekend_mask = y_time[..., 4] > 0.5
        weekday_mask = ~weekend_mask

        def add_mask(mask: torch.Tensor, sum_attr: str, count_attr: str) -> None:
            mask_count = int(per_slice_count[mask].sum().item())
            if mask_count == 0:
                return
            setattr(self, sum_attr, getattr(self, sum_attr) + float(per_slice_abs[mask].sum().item()))
            setattr(self, count_attr, getattr(self, count_attr) + mask_count)

        add_mask(peak_mask, "peak_abs_sum", "peak_count")
        add_mask(offpeak_mask, "offpeak_abs_sum", "offpeak_count")
        add_mask(weekend_mask, "weekend_abs_sum", "weekend_count")
        add_mask(weekday_mask, "weekday_abs_sum", "weekday_count")

        if self.rainy_feature_index is None:
            return
        feature_dim = int(y_time.shape[-1])
        feature_index = self.rainy_feature_index
        if feature_index < 0:
            feature_index += feature_dim
        if not 0 <= feature_index < feature_dim:
            return
        rainy_mask = y_time[..., feature_index] > self.rainy_threshold
        add_mask(rainy_mask, "rainy_abs_sum", "rainy_count")
        add_mask(~rainy_mask, "nonrainy_abs_sum", "nonrainy_count")

    def update(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        y_time: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one batch.

        Args:
            pred: Predicted OD flow, shape [B, H, N, N].
            true: Ground-truth OD flow, shape [B, H, N, N].
            y_time: Optional target time features, shape [B, H, F].
        """
        pred = pred.detach().float().cpu()
        true = true.detach().float().cpu()
        if y_time is not None:
            y_time = y_time.detach().float().cpu()
        valid = _expand_valid_mask(valid_mask, true)

        diff = pred - true
        abs_err = diff.abs()
        sq_err = diff.pow(2)

        self.abs_sum += float(abs_err[valid].sum().item())
        self.sq_sum += float(sq_err[valid].sum().item())
        self.true_abs_sum += float(true.abs()[valid].sum().item())
        self.count += int(valid.sum().item())

        nonzero_mask = (true > self.zero_threshold) & valid
        nonzero_count = int(nonzero_mask.sum().item())
        if nonzero_count > 0:
            self.nonzero_abs_sum += float(abs_err[nonzero_mask].sum().item())
            self.nonzero_count += nonzero_count

        flat_true = true.reshape(-1, true.shape[-1] * true.shape[-2])
        flat_pred = pred.reshape_as(flat_true)
        flat_valid = valid.reshape_as(flat_true)
        k = min(self.topk, flat_true.shape[-1])
        if k > 0 and flat_valid.any():
            masked_true = flat_true.masked_fill(~flat_valid, float("-inf"))
            topk_idx = torch.topk(masked_true, k=k, dim=-1).indices
            topk_true = torch.gather(flat_true, dim=-1, index=topk_idx)
            topk_pred = torch.gather(flat_pred, dim=-1, index=topk_idx)
            topk_valid = torch.gather(flat_valid, dim=-1, index=topk_idx)
            topk_abs = (topk_pred - topk_true).abs()
            if topk_valid.any():
                self.topk_abs_sum += float(topk_abs[topk_valid].sum().item())
                self.topk_count += int(topk_valid.sum().item())

        positive_true = true[(true > self.zero_threshold) & valid]
        if positive_true.numel() > 0:
            threshold_source = positive_true
            if self.high_flow_max_samples > 0 and positive_true.numel() > self.high_flow_max_samples:
                sample_idx = torch.linspace(
                    0,
                    positive_true.numel() - 1,
                    steps=self.high_flow_max_samples,
                    dtype=torch.long,
                )
                threshold_source = positive_true[sample_idx]
            q = min(max(self.high_flow_quantile, 0.0), 1.0)
            kth = max(1, min(threshold_source.numel(), int(math.ceil(q * threshold_source.numel()))))
            threshold = torch.kthvalue(threshold_source, kth).values
            high_mask = (true >= threshold) & valid
            high_count = int(high_mask.sum().item())
            if high_count > 0:
                self.high_flow_abs_sum += float(abs_err[high_mask].sum().item())
                self.high_flow_count += high_count

        zero_mask = (true <= self.zero_threshold) & valid
        zero_count = int(zero_mask.sum().item())
        if zero_count > 0:
            self.zero_fp_count += int(((pred > self.pred_positive_threshold) & zero_mask).sum().item())
            self.zero_count += zero_count

        if y_time is not None:
            self._update_time_mask(abs_err, y_time, valid_mask=valid)

    def compute(self) -> dict[str, float]:
        metrics = {
            "mae": self._ratio(self.abs_sum, self.count),
            "rmse": math.sqrt(self._ratio(self.sq_sum, self.count)),
            "wape": float(self.abs_sum / max(self.true_abs_sum, 1e-8)),
            "nonzero_mae": self._ratio(self.nonzero_abs_sum, self.nonzero_count),
            "topk_mae": self._ratio(self.topk_abs_sum, self.topk_count),
            "high_flow_mae": self._ratio(self.high_flow_abs_sum, self.high_flow_count),
            "zero_false_positive_rate": self._ratio(self.zero_fp_count, self.zero_count),
        }
        if self.peak_count + self.offpeak_count > 0:
            metrics.update(
                {
                    "peak_hour_mae": self._ratio(self.peak_abs_sum, self.peak_count),
                    "offpeak_mae": self._ratio(self.offpeak_abs_sum, self.offpeak_count),
                    "weekend_mae": self._ratio(self.weekend_abs_sum, self.weekend_count),
                    "weekday_mae": self._ratio(self.weekday_abs_sum, self.weekday_count),
                }
            )
        if self.rainy_count + self.nonrainy_count > 0:
            metrics.update(
                {
                    "rainy_mae": self._ratio(self.rainy_abs_sum, self.rainy_count),
                    "nonrainy_mae": self._ratio(self.nonrainy_abs_sum, self.nonrainy_count),
                }
            )
        return _filter_report_keys(metrics, self.report_keys)


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float) and math.isfinite(value):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
