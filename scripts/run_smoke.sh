#!/usr/bin/env bash
set -euo pipefail

SKIP_CUDA_CHECK="${SKIP_CUDA_CHECK:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-cuda-check)
      SKIP_CUDA_CHECK=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"

if [[ "$SKIP_CUDA_CHECK" != "1" ]]; then
  uv run --no-sync python scripts/check_cuda.py
fi

if [[ ! -f "data/toy/od.npy" ]]; then
  uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy
fi

uv run --no-sync python src/train.py --config configs/default.yaml
uv run --no-sync python src/train.py --config configs/od_llm_tiny.yaml
uv run --no-sync python src/compare.py --root outputs --output outputs/comparison.csv

echo ""
echo "Smoke run finished."
echo "Metrics table: outputs/comparison.csv"
