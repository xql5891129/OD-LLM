from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import deep_update, load_config  # noqa: E402


def write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of OD-LLM experiments from a suite yaml.")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    with suite_path.open("r", encoding="utf-8") as f:
        suite = yaml.safe_load(f) or {}

    base_config_path = Path(args.base_config or suite.get("base_config", "configs/default.yaml"))
    base_cfg = load_config(base_config_path)
    suite_name = suite.get("suite_name", suite_path.stem)
    generated_dir = ROOT / "outputs" / "_generated_configs" / suite_name
    selected = set(args.only or [])

    experiments = suite.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {suite_path}")

    for exp in experiments:
        name = exp["name"]
        if selected and name not in selected:
            continue
        cfg = deep_update(base_cfg, {k: v for k, v in exp.items() if k != "name"})
        cfg.setdefault("experiment", {})["name"] = name
        config_path = generated_dir / f"{name}.yaml"
        write_config(cfg, config_path)
        run_command([args.python, "src/train.py", "--config", str(config_path)], dry_run=args.dry_run)

    if not args.skip_compare:
        output = ROOT / "outputs" / f"{suite_name}_comparison.csv"
        run_command(
            [args.python, "src/compare.py", "--root", "outputs", "--output", str(output)],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
