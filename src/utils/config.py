from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("inherits", None)
    if parent:
        parent_path = path.parent / parent
        base = load_config(parent_path)
        cfg = deep_update(base, cfg)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    outputs = cfg.get("outputs", {})
    for key in ["root", "checkpoints", "logs", "figures", "explanations"]:
        if key in outputs:
            Path(outputs[key]).mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(cfg: dict[str, Any]) -> None:
    """Resolve per-run output directories in-place."""
    outputs = cfg.setdefault("outputs", {})
    root = Path(outputs.get("root", "outputs"))
    experiment_name = cfg.get("experiment", {}).get("name")
    if not experiment_name:
        experiment_name = cfg.get("model", {}).get("name", "run")

    if outputs.get("use_run_subdir", True):
        run_dir = root / experiment_name
        outputs["run_dir"] = str(run_dir)
        outputs["checkpoints"] = str(run_dir / "checkpoints")
        outputs["logs"] = str(run_dir / "logs")
        outputs["figures"] = str(run_dir / "figures")
        outputs["explanations"] = str(run_dir / "explanations")
    else:
        outputs.setdefault("run_dir", str(root))
        outputs.setdefault("checkpoints", str(root / "checkpoints"))
        outputs.setdefault("logs", str(root / "logs"))
        outputs.setdefault("figures", str(root / "figures"))
        outputs.setdefault("explanations", str(root / "explanations"))

    ensure_dirs(cfg)
