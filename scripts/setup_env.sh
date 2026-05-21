#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5-1.5b}"
PROVIDER="${PROVIDER:-huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-hf_models}"
LLM_LAYERS="${LLM_LAYERS:-6}"
SKIP_MODEL="${SKIP_MODEL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --llm-layers)
      LLM_LAYERS="$2"
      shift 2
      ;;
    --skip-model)
      SKIP_MODEL=1
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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
echo "Using project root: $PROJECT_ROOT"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

echo "Syncing Python environment with uv..."
uv sync --extra download

if [[ "$SKIP_MODEL" != "1" ]]; then
  echo "Downloading model '$MODEL' from '$PROVIDER'..."
  uv run python scripts/download_model.py \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --output-dir "$OUTPUT_DIR" \
    --llm-layers "$LLM_LAYERS"
else
  echo "Skipping model download."
fi

echo "Running a quick compile check..."
uv run python -m compileall src

echo ""
echo "Setup finished."
echo "Try a smoke test:"
echo "  uv run python scripts/generate_toy_data.py --output data/toy/od.npy"
echo "  uv run python src/train.py --config configs/od_llm_tiny.yaml"

