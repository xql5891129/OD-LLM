from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json


TIME_COLUMNS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend"]


@dataclass
class ODTransform:
    name: str = "none"

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.name == "none":
            return array
        if self.name == "log1p":
            return np.log1p(array)
        raise ValueError(f"Unknown transform: {self.name}")

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.name == "none":
            return tensor
        if self.name == "log1p":
            return torch.expm1(tensor).clamp_min(0.0)
        raise ValueError(f"Unknown transform: {self.name}")


def _time_features_from_index(length: int) -> np.ndarray:
    idx = np.arange(length, dtype=np.float32)
    phase_day = 2 * np.pi * (idx % 48) / 48.0
    phase_week = 2 * np.pi * (idx % (48 * 7)) / float(48 * 7)
    return np.stack(
        [
            np.sin(phase_day),
            np.cos(phase_day),
            np.sin(phase_week),
            np.cos(phase_week),
            np.zeros_like(idx),
        ],
        axis=-1,
    ).astype(np.float32)


def _time_features_from_datetimes(times: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(times)
    minute_of_day = dt.dt.hour.to_numpy() * 60 + dt.dt.minute.to_numpy()
    day_phase = 2 * np.pi * minute_of_day / 1440.0
    dow = dt.dt.dayofweek.to_numpy()
    week_phase = 2 * np.pi * dow / 7.0
    is_weekend = (dow >= 5).astype(np.float32)
    return np.stack(
        [
            np.sin(day_phase),
            np.cos(day_phase),
            np.sin(week_phase),
            np.cos(week_phase),
            is_weekend,
        ],
        axis=-1,
    ).astype(np.float32)


def _select_time_features(
    time_features: np.ndarray,
    columns: list[str],
    mode: str,
    selected_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Select time/context features by experiment mode.

    The first five columns are always calendar features:
    tod_sin, tod_cos, dow_sin, dow_cos, is_weekend.
    """
    mode = mode.lower()
    if mode in {"all", "full"}:
        return time_features, columns
    if mode in {"calendar", "time", "time_only"}:
        return time_features[:, :5], columns[:5]

    if mode in {"core_weather", "weather_core"}:
        keep_names = {
            "weather_temperature_c_z",
            "weather_precip_mm_z",
            "weather_wind_speed_ms_z",
            "weather_is_rainy",
        }
        selected_columns = [*columns[:5], *[col for col in columns[5:] if col in keep_names]]
    elif mode in {"selected", "custom"}:
        if not selected_columns:
            raise ValueError("data.time_features_mode=selected requires data.time_feature_columns.")
        selected_columns = selected_columns
    else:
        raise ValueError(f"Unsupported data.time_features_mode: {mode}")

    column_to_idx = {name: idx for idx, name in enumerate(columns)}
    missing = [name for name in selected_columns if name not in column_to_idx]
    if missing:
        raise ValueError(f"Selected time feature columns not found: {missing}")
    indices = [column_to_idx[name] for name in selected_columns]
    return time_features[:, indices], list(selected_columns)


def load_od_array(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fmt = cfg["format"].lower()
    path = Path(cfg["path"])
    meta: dict[str, Any] = {}

    if fmt == "npy":
        od = np.load(path, mmap_mode=cfg.get("mmap_mode")).astype(np.float32, copy=False)
        if od.ndim != 3 or od.shape[1] != od.shape[2]:
            raise ValueError(f"Expected npy shape [T, N, N], got {od.shape}")
        time_features_path = cfg.get("time_features_path")
        if time_features_path is None:
            candidate = path.parent / "time_features.npy"
            time_features_path = candidate if candidate.exists() else None
        if time_features_path is not None:
            time_features = np.load(time_features_path).astype(np.float32)
            if time_features.shape[0] != od.shape[0]:
                raise ValueError(
                    f"time_features length {time_features.shape[0]} does not match OD length {od.shape[0]}"
                )
            columns_path = Path(time_features_path).with_name("time_features_columns.json")
            if columns_path.exists():
                with columns_path.open("r", encoding="utf-8") as f:
                    meta["time_features_columns"] = json.load(f)
            columns = meta.get("time_features_columns") or [f"feature_{idx}" for idx in range(time_features.shape[1])]
            time_features, columns = _select_time_features(
                time_features,
                columns,
                mode=str(cfg.get("time_features_mode", "all")),
                selected_columns=cfg.get("time_feature_columns"),
            )
            meta["time_features_columns"] = columns
        else:
            time_features = _time_features_from_index(od.shape[0])
            meta["time_features_columns"] = list(TIME_COLUMNS)
        times_path = cfg.get("times_path")
        if times_path is None:
            candidate = path.parent / "times.csv"
            times_path = candidate if candidate.exists() else None
        if times_path is not None:
            time_df = pd.read_csv(times_path)
            meta["times"] = time_df["time"].astype(str).tolist() if "time" in time_df.columns else None
        else:
            meta["times"] = None
        return od, time_features, meta

    if fmt == "csv":
        csv_cfg = cfg.get("csv", {})
        time_col = csv_cfg.get("time_col", "time")
        origin_col = csv_cfg.get("origin_col", "origin")
        dest_col = csv_cfg.get("destination_col", "destination")
        flow_col = csv_cfg.get("flow_col", "flow")
        df = pd.read_csv(path)
        for col in [time_col, origin_col, dest_col, flow_col]:
            if col not in df.columns:
                raise ValueError(f"Missing CSV column: {col}")

        times = pd.Series(sorted(pd.to_datetime(df[time_col]).unique()))
        origins = sorted(pd.unique(pd.concat([df[origin_col], df[dest_col]], ignore_index=True)))
        station_to_idx = {station: i for i, station in enumerate(origins)}
        time_to_idx = {time: i for i, time in enumerate(times)}

        od = np.zeros((len(times), len(origins), len(origins)), dtype=np.float32)
        for row in df.itertuples(index=False):
            t = getattr(row, time_col)
            o = getattr(row, origin_col)
            d = getattr(row, dest_col)
            flow = getattr(row, flow_col)
            od[time_to_idx[pd.Timestamp(t)], station_to_idx[o], station_to_idx[d]] += float(flow)

        time_features = _time_features_from_datetimes(times)
        meta["times"] = [str(t) for t in times]
        meta["station_to_idx"] = station_to_idx
        return od, time_features, meta

    raise ValueError(f"Unsupported data format: {fmt}")


def load_poi_features(cfg: dict[str, Any], num_nodes: int) -> tuple[np.ndarray | None, list[str]]:
    """Load optional station-level POI/static features.

    Returns:
        poi_features: [N, F] float32, or None when absent/disabled.
        columns: POI feature column names if available.
    """
    if not bool(cfg.get("use_poi_features", False)):
        return None, []

    path = cfg.get("poi_features_path")
    if path is None:
        od_path = Path(cfg["path"])
        candidate = od_path.parent / "poi_features.npy"
        path = candidate if candidate.exists() else None
    if path is None:
        raise FileNotFoundError("data.use_poi_features=true but poi_features.npy was not found.")

    poi = np.load(path).astype(np.float32, copy=False)
    if poi.ndim != 2:
        raise ValueError(f"Expected poi features shape [N,F], got {poi.shape}")
    if poi.shape[0] != num_nodes:
        raise ValueError(f"POI feature node count {poi.shape[0]} does not match OD N={num_nodes}")

    columns: list[str] = []
    columns_path = Path(path).with_name("poi_feature_columns.json")
    if columns_path.exists():
        with columns_path.open("r", encoding="utf-8") as f:
            columns = json.load(f)
    return poi, columns


class ODDataset(Dataset):
    """Chronological OD sliding-window dataset.

    Returns:
        x: [L, N, N]
        y: [H, N, N]
        x_time: [L, F]
        y_time: [H, F]
    """

    def __init__(self, cfg: dict[str, Any], split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.input_len = int(cfg["input_len"])
        self.pred_len = int(cfg["pred_len"])
        self.prev_lag = int(cfg.get("prev_lag", 0))
        self.transform = ODTransform(cfg.get("transform", "none"))

        od_raw, time_features, meta = load_od_array(cfg)
        self.raw_od = od_raw
        self.od = self.transform.transform(od_raw).astype(np.float32, copy=False)
        self.time_features = time_features.astype(np.float32)
        self.meta = meta
        self.num_nodes = int(self.od.shape[1])
        poi_features, poi_columns = load_poi_features(cfg, self.num_nodes)
        self.poi_features = poi_features
        self.poi_feature_dim = 0 if poi_features is None else int(poi_features.shape[1])
        self.meta["poi_feature_columns"] = poi_columns
        self.total_steps = int(self.od.shape[0])

        train_ratio = float(cfg.get("train_ratio", 0.7))
        val_ratio = float(cfg.get("val_ratio", 0.1))
        train_end = int(self.total_steps * train_ratio)
        val_end = int(self.total_steps * (train_ratio + val_ratio))

        max_start = self.total_steps - self.input_len - self.pred_len + 1
        if max_start <= 0:
            raise ValueError("Time series is too short for input_len + pred_len")

        starts: list[int] = []
        for start in range(max_start):
            target_start = start + self.input_len
            if split == "train" and target_start < train_end:
                starts.append(start)
            elif split == "val" and train_end <= target_start < val_end:
                starts.append(start)
            elif split == "test" and target_start >= val_end:
                starts.append(start)
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.starts[index]
        x_start = start
        x_end = start + self.input_len
        y_start = x_end
        y_end = y_start + self.pred_len
        prev_x_start = x_start - self.prev_lag
        prev_y_start = y_start - self.prev_lag
        if self.prev_lag > 0 and prev_x_start >= 0 and prev_y_start >= 0:
            prev_x = self.od[prev_x_start : prev_x_start + self.input_len]
            prev_y = self.od[prev_y_start : prev_y_start + self.pred_len]
            prev_valid = True
        else:
            prev_x = self.od[x_start:x_end]
            prev_y = self.od[y_start:y_end]
            prev_valid = False

        return {
            "x": torch.from_numpy(self.od[x_start:x_end]),          # [L, N, N]
            "y": torch.from_numpy(self.od[y_start:y_end]),          # [H, N, N]
            "prev_x": torch.from_numpy(prev_x),                     # [L, N, N]
            "prev_y": torch.from_numpy(prev_y),                     # [H, N, N]
            "x_time": torch.from_numpy(self.time_features[x_start:x_end]),
            "y_time": torch.from_numpy(self.time_features[y_start:y_end]),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "target_start": torch.tensor(y_start, dtype=torch.long),
            "prev_valid": torch.tensor(prev_valid, dtype=torch.bool),
        }

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform.inverse_transform(tensor)

