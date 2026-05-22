#!/usr/bin/env bash
set -euo pipefail

TRANSFORMER_CONFIG="configs/real_od_transformer.yaml"
ODLLM_CONFIG="configs/real_od_qwen.yaml"
BASELINE_SUITE="configs/suite_baselines.yaml"
ABLATION_SUITE="configs/suite_ablation.yaml"
SKIP_CUDA_CHECK=0
SKIP_BASELINES=0
SKIP_ABLATIONS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transformer-config)
      TRANSFORMER_CONFIG="$2"
      shift 2
      ;;
    --od-llm-config)
      ODLLM_CONFIG="$2"
      shift 2
      ;;
    --baseline-suite)
      BASELINE_SUITE="$2"
      shift 2
      ;;
    --ablation-suite)
      ABLATION_SUITE="$2"
      shift 2
      ;;
    --skip-cuda-check)
      SKIP_CUDA_CHECK=1
      shift
      ;;
    --skip-baselines)
      SKIP_BASELINES=1
      shift
      ;;
    --skip-ablations)
      SKIP_ABLATIONS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ "$DRY_RUN" == "1" ]]; then
  echo "uv run --no-sync python src/train.py --config $TRANSFORMER_CONFIG"
  echo "uv run --no-sync python src/train.py --config $ODLLM_CONFIG"
else
  uv run --no-sync python src/train.py --config "$TRANSFORMER_CONFIG"
  uv run --no-sync python src/train.py --config "$ODLLM_CONFIG"
fi

if [[ "$SKIP_BASELINES" != "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    uv run --no-sync python scripts/run_experiment_suite.py --suite "$BASELINE_SUITE" --dry-run
  else
    uv run --no-sync python scripts/run_experiment_suite.py --suite "$BASELINE_SUITE"
  fi
fi

if [[ "$SKIP_ABLATIONS" != "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    uv run --no-sync python scripts/run_experiment_suite.py --suite "$ABLATION_SUITE" --dry-run
  else
    uv run --no-sync python scripts/run_experiment_suite.py --suite "$ABLATION_SUITE"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  uv run --no-sync python src/compare.py --root outputs --output outputs/paper_experiments_comparison.csv
  echo "Saved final comparison: outputs/paper_experiments_comparison.csv"
fi
