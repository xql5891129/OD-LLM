#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5-1.5b}"
PROVIDER="${PROVIDER:-huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-hf_models}"
LLM_LAYERS="${LLM_LAYERS:-6}"
SKIP_MODEL="${SKIP_MODEL:-0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"

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
    --torch-backend)
      TORCH_BACKEND="$2"
      shift 2
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
uv sync --extra download --extra data

case "$TORCH_BACKEND" in
  skip)
    echo "Skipping explicit PyTorch backend installation."
    ;;
  cpu)
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    ;;
  cu128)
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    ;;
  cu130)
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    ;;
  *)
    echo "Unknown --torch-backend: $TORCH_BACKEND. Use one of: skip, cpu, cu128, cu130."
    exit 1
    ;;
esac

if [[ "$TORCH_BACKEND" != "skip" ]]; then
  echo "Installing PyTorch backend '$TORCH_BACKEND' from $TORCH_INDEX ..."
  uv pip install --reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

if [[ "$SKIP_MODEL" != "1" ]]; then
  echo "Downloading model '$MODEL' from '$PROVIDER'..."
  uv run --no-sync python scripts/download_model.py \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --output-dir "$OUTPUT_DIR" \
    --llm-layers "$LLM_LAYERS"
else
  echo "Skipping model download."
fi

echo "Running a quick compile check..."
uv run --no-sync python -m compileall src scripts

echo "Checking PyTorch CUDA visibility..."
uv run --no-sync python scripts/check_cuda.py

echo ""
echo "Setup finished."
echo "Try a smoke test:"
echo "  uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy"
echo "  uv run --no-sync python src/train.py --config configs/od_llm_tiny.yaml"
