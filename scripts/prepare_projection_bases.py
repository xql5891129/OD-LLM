from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.line_dataset import LineODDataset  # noqa: E402
from utils.config import load_config  # noqa: E402


def stable_eigenvectors(vectors: torch.Tensor) -> torch.Tensor:
    """Resolve arbitrary eigenvector signs for reproducible saved bases."""
    columns = torch.arange(vectors.shape[1], device=vectors.device)
    pivot_rows = vectors.abs().argmax(dim=0)
    signs = torch.sign(vectors[pivot_rows, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return vectors * signs.unsqueeze(0)


def top_eigenvectors(covariance: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    selected_values = eigenvalues[order[:rank]].clamp_min(0.0)
    selected_vectors = stable_eigenvectors(eigenvectors[:, order[:rank]])
    total_energy = eigenvalues.clamp_min(0.0).sum().item()
    retained = selected_values.sum().item() / total_energy if total_energy > 0 else 0.0
    return selected_vectors, retained


def accumulate_two_sided_covariances(
    dataset: LineODDataset,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate train-only origin/destination covariances on valid OD cells."""
    nmax = dataset.num_nodes
    origin_cov = torch.zeros((nmax, nmax), dtype=torch.float64)
    destination_cov = torch.zeros((nmax, nmax), dtype=torch.float64)
    train_ratio = float(dataset.cfg.get("train_ratio", 0.7))
    used_steps = 0

    for entry in dataset.line_entries:
        od = entry["od"]
        num_steps = int(od.shape[0])
        train_end = int(num_steps * train_ratio)
        num_stops = int(entry["num_stops"])
        valid_mask = torch.from_numpy(np.asarray(entry["mask"], dtype=np.float64))
        print(f"Accumulating {entry['line_dir']}: train_steps={train_end}, stops={num_stops}")

        for start in range(0, train_end, chunk_size):
            stop = min(start + chunk_size, train_end)
            chunk = np.asarray(od[start:stop], dtype=np.float64)
            x = torch.from_numpy(chunk.copy()) * valid_mask
            origin_cov[:num_stops, :num_stops] += torch.einsum("tij,tkj->ik", x, x)
            destination_cov[:num_stops, :num_stops] += torch.einsum("tij,tik->jk", x, x)
            used_steps += int(x.shape[0])

    return origin_cov, destination_cov, used_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train-only two-sided SVD bases for OD tensor tokenizer ablations."
    )
    parser.add_argument("--config", required=True, help="Line-bus base yaml configuration.")
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    if args.rank < 1:
        raise ValueError("--rank must be positive")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    config_path = Path(args.config)
    cfg = load_config(config_path)
    if str(cfg.get("data", {}).get("dataset_type", "")).lower() not in {"line", "line_bus", "bus_line"}:
        raise ValueError("This script only supports data.dataset_type=line_bus.")
    dataset = LineODDataset(cfg["data"], split="train")
    if args.rank > dataset.num_nodes:
        raise ValueError(f"rank={args.rank} exceeds Nmax={dataset.num_nodes}")

    origin_cov, destination_cov, used_steps = accumulate_two_sided_covariances(dataset, args.chunk_size)
    po, origin_energy = top_eigenvectors(origin_cov, args.rank)
    pd, destination_energy = top_eigenvectors(destination_cov, args.rank)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Po": po.float().contiguous(),
        "Pd": pd.float().contiguous(),
        "metadata": {
            "method": "train_only_two_sided_svd",
            "config": str(config_path),
            "rank": int(args.rank),
            "num_nodes": int(dataset.num_nodes),
            "num_lines": int(dataset.num_lines),
            "train_ratio": float(dataset.cfg.get("train_ratio", 0.7)),
            "used_time_steps": int(used_steps),
            "origin_energy_ratio": float(origin_energy),
            "destination_energy_ratio": float(destination_energy),
        },
    }
    torch.save(payload, output)
    print(
        f"Saved two-sided SVD bases: {output} | "
        f"Po={tuple(po.shape)}, Pd={tuple(pd.shape)} | "
        f"energy(origin/destination)={origin_energy:.4f}/{destination_energy:.4f}"
    )


if __name__ == "__main__":
    main()
