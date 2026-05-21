from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utils.config import load_config, prepare_output_dirs


def build_explanation_prompt(evidence: dict) -> str:
    """Build a structured explanation prompt without calling an LLM API."""
    return (
        "You are a bus dispatch analyst. Explain the OD forecast using the structured evidence below.\n\n"
        f"{json.dumps(evidence, indent=2)}\n\n"
        "Please discuss total demand change, top increasing OD pairs, top decreasing OD pairs, "
        "latent mobility pattern, and possible operational implications."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--evidence", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_output_dirs(cfg)
    evidence = {"note": "Stage-5 evidence extraction is not implemented yet."}
    if args.evidence:
        with open(args.evidence, "r", encoding="utf-8") as f:
            evidence = json.load(f)

    prompt = build_explanation_prompt(evidence)
    output = Path(args.output or Path(cfg["outputs"]["explanations"]) / "prompt.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prompt, encoding="utf-8")
    print(f"Saved explanation prompt to {output}")


if __name__ == "__main__":
    main()
