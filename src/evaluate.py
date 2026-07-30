from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train import (
    build_dataloader,
    build_dataset,
    build_model,
    describe_device,
    resolve_amp_dtype,
    resolve_model_time_feature_indices,
    run_epoch,
)
from utils.config import load_config, prepare_output_dirs
from utils.metrics import format_metrics
from utils.seed import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/common/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for evaluation only.")
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Evaluate at most this many batches. Use only for quick diagnostics.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default=None,
        help="Optional metrics filename under outputs.logs. Defaults to test_metrics.json and evaluate_metrics.json.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.max_eval_batches is not None:
        cfg.setdefault("train", {})["max_eval_batches_per_epoch"] = int(args.max_eval_batches)
    prepare_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    describe_device(device)

    test_set = build_dataset(cfg["data"], "test")
    test_loader = build_dataloader(test_set, cfg, shuffle=False, device=device)
    dataset_time_feature_dim = int(test_set.time_features.shape[1]) if test_set.time_features.ndim == 2 else 0
    model_time_feature_indices = resolve_model_time_feature_indices(
        cfg,
        getattr(test_set, "time_feature_columns", None),
        dataset_time_feature_dim,
    )
    cfg.setdefault("_runtime", {})["model_time_feature_indices"] = model_time_feature_indices
    time_feature_dim = (
        dataset_time_feature_dim
        if model_time_feature_indices is None
        else len(model_time_feature_indices)
    )
    model = build_model(
        cfg,
        test_set.num_nodes,
        time_feature_dim=time_feature_dim,
        poi_features=test_set.poi_features,
        poi_feature_dim=test_set.poi_feature_dim,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    amp_dtype = resolve_amp_dtype(cfg["train"].get("amp_dtype", "bf16"))
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        cfg=cfg,
        train=False,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    print(f"Test loss={test_loss:.4f} | {format_metrics(test_metrics)}")
    logs_dir = Path(cfg["outputs"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"test_loss": test_loss, "test_metrics": test_metrics}
    filenames = [args.metrics_name] if args.metrics_name else ["test_metrics.json", "evaluate_metrics.json"]
    for filename in filenames:
        out_path = logs_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved test metrics to {out_path}")


if __name__ == "__main__":
    main()
