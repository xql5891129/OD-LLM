from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ODDataset
from losses.sparse_od_loss import sparse_od_loss
from models.od_llm import ODLLM
from models.transformer_baseline import ODTensorTransformer
from utils.config import load_config, prepare_output_dirs
from utils.metrics import compute_metrics, format_metrics
from utils.seed import get_device, set_seed


def build_model(cfg: dict, num_nodes: int) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "od_tensor_transformer")
    common = dict(
        num_nodes=num_nodes,
        input_len=int(data_cfg["input_len"]),
        pred_len=int(data_cfg["pred_len"]),
        rank=int(model_cfg["rank"]),
        d_model=int(model_cfg["d_model"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        max_tokens=int(model_cfg.get("max_tokens", 4096)),
    )
    if name == "od_tensor_transformer":
        return ODTensorTransformer(
            **common,
            n_heads=int(model_cfg["n_heads"]),
            num_layers=int(model_cfg["num_layers"]),
        )
    if name in {"od_llm", "od_llm_gpt2"}:
        return ODLLM(
            **common,
            llm_model=str(model_cfg.get("llm_model", "mini_gpt")),
            llm_dim=int(model_cfg.get("llm_dim", 768)),
            llm_layers=int(model_cfg.get("llm_layers", 6)),
            llm_heads=int(model_cfg.get("llm_heads", model_cfg.get("n_heads", 8))),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_path=model_cfg.get("pretrained_path"),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            freeze_llm=bool(model_cfg.get("freeze_llm", True)),
            use_reprogramming=bool(model_cfg.get("use_reprogramming", False)),
            num_virtual_prompt_tokens=int(model_cfg.get("num_virtual_prompt_tokens", 8)),
            num_source_tokens=int(model_cfg.get("num_source_tokens", 1000)),
        )
    raise NotImplementedError(f"Model is not implemented: {name}")


def run_epoch(model, loader, optimizer, device, cfg, train: bool) -> tuple[float, dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    trues = []
    loss_cfg = cfg["loss"]

    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        x = batch["x"].float().to(device)  # [B, L, N, N]
        y = batch["y"].float().to(device)  # [B, H, N, N]

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss, parts = sparse_od_loss(
                pred,
                y,
                alpha=float(loss_cfg["alpha"]),
                beta=float(loss_cfg["beta"]),
                gamma=float(loss_cfg["gamma"]),
                topk=int(loss_cfg["topk"]),
                nonzero_threshold=float(loss_cfg.get("nonzero_threshold", 0.0)),
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = cfg["train"].get("grad_clip")
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        preds.append(pred.detach().cpu())
        trues.append(y.detach().cpu())
        iterator.set_postfix(loss=f"{losses[-1]:.4f}", **parts)

    pred_all = torch.cat(preds, dim=0)
    true_all = torch.cat(trues, dim=0)
    metrics = compute_metrics(pred_all, true_all, **cfg["metrics"])
    return sum(losses) / max(len(losses), 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    train_set = ODDataset(cfg["data"], "train")
    val_set = ODDataset(cfg["data"], "val")
    test_set = ODDataset(cfg["data"], "test")
    print(f"Dataset: N={train_set.num_nodes}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=bool(cfg["train"].get("shuffle_train", True)),
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )

    model = build_model(cfg, train_set.num_nodes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(cfg["outputs"]["checkpoints"])
    best_path = ckpt_dir / "best.pt"
    history_path = Path(cfg["outputs"]["logs"]) / "train_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, device, cfg, train=False)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
            f"val {format_metrics(val_metrics)}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg, "num_nodes": train_set.num_nodes}, best_path)
            print(f"Saved best checkpoint to {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["train"].get("patience", 5)):
                print("Early stopping")
                break

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_metrics = run_epoch(model, test_loader, optimizer=None, device=device, cfg=cfg, train=False)
    metrics_path = Path(cfg["outputs"]["logs"]) / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)
    print(f"Test loss={test_loss:.4f} | {format_metrics(test_metrics)}")
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
