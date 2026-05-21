from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_results(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(root.glob("*/logs/test_metrics.json")):
        run_name = metrics_path.parents[1].name
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row = {"run": run_name, "test_loss": payload.get("test_loss")}
        row.update(payload.get("test_metrics", {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("test_loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--output", type=str, default="outputs/comparison.csv")
    args = parser.parse_args()

    df = collect_results(Path(args.root))
    if df.empty:
        print(f"No test metrics found under {args.root}")
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print(f"Saved comparison table to {output}")


if __name__ == "__main__":
    main()

