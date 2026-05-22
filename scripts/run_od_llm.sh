#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python src/train.py --config configs/od_llm.yaml
