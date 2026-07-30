from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset import ODTransform, TIME_COLUMNS, _select_time_features, _time_features_from_index


class LineODDataset(Dataset):
    """Line-direction OD sliding-window dataset.

    Each sample comes from one bus line and one direction. Line-specific OD
    matrices are padded to Nmax, while `od_mask` marks the feasible downstream
    OD cells that should participate in loss and metrics.

    Returns:
        x: [L, Nmax, Nmax]
        y: [H, Nmax, Nmax]
        od_mask: [Nmax, Nmax], valid same-line downstream OD pairs
        node_mask: [Nmax], real stops before padding
        x_time: [L, F]
        y_time: [H, F]
        poi_features: [Nmax, Fp] when data.use_poi_features=true
    """

    def __init__(self, cfg: dict[str, Any], split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["root"])
        if not self.root.exists():
            raise FileNotFoundError(f"Line bus dataset root not found: {self.root}")

        self.input_len = int(cfg["input_len"])
        self.pred_len = int(cfg["pred_len"])
        self.prev_lag = int(cfg.get("prev_lag", 0))
        self.sample_stride = max(int(cfg.get("sample_stride", 1)), 1)
        self.transform = ODTransform(cfg.get("transform", "none"))
        self.return_poi = bool(cfg.get("use_poi_features", False))
        self.mmap_mode = cfg.get("mmap_mode")

        self.time_features, self.time_feature_columns = self._load_time_features()
        self.line_entries = self._load_line_entries()
        if not self.line_entries:
            raise ValueError(f"No usable line OD folders found under {self.root}")

        configured_n = cfg.get("max_nodes", "auto")
        if configured_n in {None, "auto"}:
            self.num_nodes = max(entry["num_stops"] for entry in self.line_entries)
        else:
            self.num_nodes = int(configured_n)
        if self.num_nodes <= 1:
            raise ValueError("LineODDataset requires max_nodes > 1")

        self.poi_feature_dim = self._infer_poi_feature_dim()
        self.poi_features = None
        self.num_lines = len(self.line_entries)
        self.samples: list[tuple[int, int]] = []
        self._build_samples()
        if not self.samples:
            raise ValueError(f"No {split} samples for line bus dataset: {self.root}")

    def _load_time_features(self) -> tuple[np.ndarray, list[str]]:
        path = self.root / "time_features.npy"
        if path.exists():
            time_features = np.load(path).astype(np.float32, copy=False)
            columns_path = self.root / "time_features_columns.json"
            if columns_path.exists():
                with columns_path.open("r", encoding="utf-8") as f:
                    columns = json.load(f)
            else:
                columns = [f"feature_{idx}" for idx in range(time_features.shape[1])]
            selected, selected_columns = _select_time_features(
                time_features,
                columns,
                mode=str(self.cfg.get("time_features_mode", "all")),
                selected_columns=self.cfg.get("time_feature_columns"),
            )
            return selected.astype(np.float32, copy=False), selected_columns

        metadata_path = self.root / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            length = int(metadata["num_time_steps"])
        else:
            first_od = next(self.root.glob("line_*_dir*/od.npy"), None)
            if first_od is None:
                raise FileNotFoundError(f"Missing time_features.npy and line OD files under {self.root}")
            length = int(np.load(first_od, mmap_mode="r").shape[0])
        return _time_features_from_index(length), list(TIME_COLUMNS)

    def _load_line_entries(self) -> list[dict[str, Any]]:
        index_path = self.root / "line_index.csv"
        if index_path.exists():
            index_df = pd.read_csv(index_path)
        else:
            rows = []
            for path in sorted(self.root.glob("line_*_dir*")):
                if (path / "od.npy").exists():
                    rows.append({"line_dir": path.name, "relative_dir": path.name, "enabled": True})
            index_df = pd.DataFrame(rows)

        if index_df.empty:
            return []
        if bool(self.cfg.get("enabled_only", True)) and "enabled" in index_df.columns:
            index_df = index_df[index_df["enabled"].astype(bool)].copy()

        line_ids = self.cfg.get("line_ids")
        if line_ids:
            selected = {str(item) for item in line_ids}
            keep = index_df["line_dir"].astype(str).isin(selected)
            if "line_no" in index_df.columns:
                keep = keep | index_df["line_no"].astype(str).isin(selected)
            index_df = index_df[keep].copy()

        if "total_flow" in index_df.columns:
            index_df = index_df.sort_values("total_flow", ascending=False)
        max_lines = int(self.cfg.get("max_lines", 0))
        if max_lines > 0:
            index_df = index_df.head(max_lines).copy()

        entries: list[dict[str, Any]] = []
        for line_idx, row in index_df.reset_index(drop=True).iterrows():
            rel = str(row.get("relative_dir") or row.get("line_dir"))
            line_dir = self.root / rel
            od_path = line_dir / "od.npy"
            mask_path = line_dir / "mask.npy"
            if not od_path.exists() or not mask_path.exists():
                continue
            od = np.load(od_path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)
            if od.ndim != 3 or od.shape[1] != od.shape[2]:
                raise ValueError(f"Expected [T,S,S] OD at {od_path}, got {od.shape}")
            mask = np.load(mask_path).astype(bool, copy=False)
            if mask.shape != od.shape[1:]:
                raise ValueError(f"mask shape {mask.shape} does not match OD shape {od.shape[1:]} at {mask_path}")
            entries.append(
                {
                    "line_index": int(line_idx),
                    "line_dir": line_dir.name,
                    "line_no": str(row.get("line_no", "")),
                    "direction": int(row.get("direction", -1)) if pd.notna(row.get("direction", -1)) else -1,
                    "num_stops": int(od.shape[1]),
                    "od": self.transform.transform(od).astype(np.float32, copy=False),
                    "mask": mask,
                    "poi_path": line_dir / "poi_features.npy",
                }
            )
        return entries

    def _infer_poi_feature_dim(self) -> int:
        if not self.return_poi:
            return 0
        for entry in self.line_entries:
            path = entry["poi_path"]
            if path.exists():
                poi = np.load(path).astype(np.float32, copy=False)
                if poi.ndim != 2:
                    raise ValueError(f"Expected [S,F] POI features at {path}, got {poi.shape}")
                return int(poi.shape[1])
        return 0

    def _build_samples(self) -> None:
        train_ratio = float(self.cfg.get("train_ratio", 0.7))
        val_ratio = float(self.cfg.get("val_ratio", 0.1))
        for line_idx, entry in enumerate(self.line_entries):
            total_steps = int(entry["od"].shape[0])
            if self.time_features.shape[0] != total_steps:
                raise ValueError(
                    f"time_features length {self.time_features.shape[0]} does not match "
                    f"{entry['line_dir']} OD length {total_steps}"
                )
            train_end = int(total_steps * train_ratio)
            val_end = int(total_steps * (train_ratio + val_ratio))
            max_start = total_steps - self.input_len - self.pred_len + 1
            if max_start <= 0:
                continue
            for start in range(0, max_start, self.sample_stride):
                target_start = start + self.input_len
                if self.split == "train" and target_start < train_end:
                    self.samples.append((line_idx, start))
                elif self.split == "val" and train_end <= target_start < val_end:
                    self.samples.append((line_idx, start))
                elif self.split == "test" and target_start >= val_end:
                    self.samples.append((line_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_od(self, value: np.ndarray) -> np.ndarray:
        padded = np.zeros((value.shape[0], self.num_nodes, self.num_nodes), dtype=np.float32)
        stops = min(value.shape[1], self.num_nodes)
        padded[:, :stops, :stops] = value[:, :stops, :stops]
        return padded

    def _pad_mask(self, mask: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        stops = min(mask.shape[0], self.num_nodes)
        padded[:stops, :stops] = mask[:stops, :stops]
        return padded

    def _load_poi(self, entry: dict[str, Any]) -> np.ndarray:
        poi = np.zeros((self.num_nodes, self.poi_feature_dim), dtype=np.float32)
        if not self.return_poi or self.poi_feature_dim <= 0 or not entry["poi_path"].exists():
            return poi
        raw = np.load(entry["poi_path"]).astype(np.float32, copy=False)
        stops = min(raw.shape[0], self.num_nodes)
        cols = min(raw.shape[1], self.poi_feature_dim)
        poi[:stops, :cols] = raw[:stops, :cols]
        return poi

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        line_idx, start = self.samples[index]
        entry = self.line_entries[line_idx]
        od = entry["od"]

        x_start = start
        x_end = start + self.input_len
        y_start = x_end
        y_end = y_start + self.pred_len
        prev_x_start = x_start - self.prev_lag
        prev_y_start = y_start - self.prev_lag
        if self.prev_lag > 0 and prev_x_start >= 0 and prev_y_start >= 0:
            prev_x = od[prev_x_start : prev_x_start + self.input_len]
            prev_y = od[prev_y_start : prev_y_start + self.pred_len]
            prev_valid = True
        else:
            prev_x = od[x_start:x_end]
            prev_y = od[y_start:y_end]
            prev_valid = False

        node_mask = np.zeros((self.num_nodes,), dtype=bool)
        node_mask[: min(entry["num_stops"], self.num_nodes)] = True
        payload: dict[str, torch.Tensor | str] = {
            "x": torch.from_numpy(self._pad_od(od[x_start:x_end])),
            "y": torch.from_numpy(self._pad_od(od[y_start:y_end])),
            "prev_x": torch.from_numpy(self._pad_od(prev_x)),
            "prev_y": torch.from_numpy(self._pad_od(prev_y)),
            "od_mask": torch.from_numpy(self._pad_mask(entry["mask"])),
            "node_mask": torch.from_numpy(node_mask),
            "x_time": torch.from_numpy(self.time_features[x_start:x_end]),
            "y_time": torch.from_numpy(self.time_features[y_start:y_end]),
            "poi_features": torch.from_numpy(self._load_poi(entry)),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "target_start": torch.tensor(y_start, dtype=torch.long),
            "prev_valid": torch.tensor(prev_valid, dtype=torch.bool),
            "line_index": torch.tensor(entry["line_index"], dtype=torch.long),
            "direction": torch.tensor(entry["direction"], dtype=torch.long),
            "num_stops": torch.tensor(entry["num_stops"], dtype=torch.long),
            "line_no": entry["line_no"],
            "line_dir": entry["line_dir"],
        }
        return payload

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform.inverse_transform(tensor)
