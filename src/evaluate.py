from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ODDataset
from train import build_model, run_epoch
from utils.config import load_config, prepare_output_dirs
from utils.metrics import format_metrics
from utils.seed import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    test_set = ODDataset(cfg["data"], "test")
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    model = build_model(cfg, test_set.num_nodes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    test_loss, test_metrics = run_epoch(model, test_loader, optimizer=None, device=device, cfg=cfg, train=False)
    print(f"Test loss={test_loss:.4f} | {format_metrics(test_metrics)}")
    out_path = Path(cfg["outputs"]["logs"]) / "evaluate_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
