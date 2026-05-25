#!/usr/bin/env bash
set -euo pipefail

SUITE="configs/suite_baselines.yaml"
BASE_CONFIG=""
DRY_RUN=0
CONTINUE_ON_ERROR=0
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --only)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ONLY+=("$1")
        shift
      done
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
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

ARGS=(run --no-sync python scripts/run_experiment_suite.py --suite "$SUITE")
if [[ -n "$BASE_CONFIG" ]]; then
  ARGS+=(--base-config "$BASE_CONFIG")
fi
if [[ ${#ONLY[@]} -gt 0 ]]; then
  ARGS+=(--only "${ONLY[@]}")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi

uv "${ARGS[@]}"
