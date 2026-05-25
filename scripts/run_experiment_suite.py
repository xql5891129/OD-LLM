from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import deep_update, load_config, prepare_output_dirs  # noqa: E402


def write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def expected_metrics_path(config: dict[str, Any]) -> Path:
    cfg = copy.deepcopy(config)
    prepare_output_dirs(cfg)
    path = Path(cfg["outputs"]["logs"]) / "test_metrics.json"
    return path if path.is_absolute() else ROOT / path


def metrics_written_after(path: Path, started_at: float) -> bool:
    if not path.exists():
        return False
    if path.stat().st_mtime < started_at:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return "test_loss" in payload and "test_metrics" in payload
    except (OSError, json.JSONDecodeError):
        return False


def run_command(command: list[str], dry_run: bool, log_path: Path | None = None) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    if log_path is None:
        completed = subprocess.run(command, cwd=ROOT)
        return int(completed.returncode)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of OD-LLM experiments from a suite yaml.")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    suite_path = Path(args.suite)
    with suite_path.open("r", encoding="utf-8") as f:
        suite = yaml.safe_load(f) or {}

    base_config_path = Path(args.base_config or suite.get("base_config", "configs/default.yaml"))
    base_cfg = load_config(base_config_path)
    suite_name = suite.get("suite_name", suite_path.stem)
    generated_dir = ROOT / "outputs" / "_generated_configs" / suite_name
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "outputs" / "suite_logs" / suite_name
    selected = set(args.only or [])
    failures: list[tuple[str, int, Path]] = []

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
        started_at = time.time()
        metrics_path = expected_metrics_path(cfg)
        log_path = log_dir / f"{name}.log"
        return_code = run_command(
            [args.python, "src/train.py", "--config", str(config_path)],
            dry_run=args.dry_run,
            log_path=log_path,
        )
        if return_code != 0:
            if metrics_written_after(metrics_path, started_at):
                print(
                    f"Experiment finished but process exited with {return_code}; "
                    f"fresh metrics found at {metrics_path}. Treating as success."
                )
                continue
            failures.append((name, return_code, log_path))
            print(f"Experiment failed: {name}, exit_code={return_code}, log={log_path}")
            if not args.continue_on_error:
                raise SystemExit(return_code)

    if not args.skip_compare:
        output = ROOT / "outputs" / f"{suite_name}_comparison.csv"
        return_code = run_command(
            [args.python, "src/compare.py", "--root", "outputs", "--output", str(output)],
            dry_run=args.dry_run,
            log_path=log_dir / "compare.log",
        )
        if return_code != 0:
            failures.append(("compare", return_code, log_dir / "compare.log"))

    if failures:
        print("Failed experiments:")
        for name, return_code, log_path in failures:
            print(f"  {name}: exit_code={return_code}, log={log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
