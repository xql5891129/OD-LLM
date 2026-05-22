from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json


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


def load_od_array(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fmt = cfg["format"].lower()
    path = Path(cfg["path"])
    meta: dict[str, Any] = {}

    if fmt == "npy":
        od = np.load(path).astype(np.float32)
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
        else:
            time_features = _time_features_from_index(od.shape[0])
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
        self.transform = ODTransform(cfg.get("transform", "none"))

        od_raw, time_features, meta = load_od_array(cfg)
        self.raw_od = od_raw
        self.od = self.transform.transform(od_raw).astype(np.float32)
        self.time_features = time_features.astype(np.float32)
        self.meta = meta
        self.num_nodes = int(self.od.shape[1])
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

        return {
            "x": torch.from_numpy(self.od[x_start:x_end]),          # [L, N, N]
            "y": torch.from_numpy(self.od[y_start:y_end]),          # [H, N, N]
            "x_time": torch.from_numpy(self.time_features[x_start:x_end]),
            "y_time": torch.from_numpy(self.time_features[y_start:y_end]),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "target_start": torch.tensor(y_start, dtype=torch.long),
        }

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform.inverse_transform(tensor)

