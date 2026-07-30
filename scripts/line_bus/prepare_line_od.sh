#!/usr/bin/env bash
set -euo pipefail

REGION="guangdianyuan"
GRANULARITY="60"
MIN_TOTAL_FLOW="100"
MIN_STOPS="4"
POI_TOP_CATEGORIES="30"
MAX_LINES="0"
MAX_CARD_FILES="0"
ALL_REGIONS=0
NO_PROGRESS=0
LINE_NOS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"
      shift 2
      ;;
    --granularity)
      GRANULARITY="$2"
      shift 2
      ;;
    --min-total-flow)
      MIN_TOTAL_FLOW="$2"
      shift 2
      ;;
    --min-stops)
      MIN_STOPS="$2"
      shift 2
      ;;
    --poi-top-categories)
      POI_TOP_CATEGORIES="$2"
      shift 2
      ;;
    --max-lines)
      MAX_LINES="$2"
      shift 2
      ;;
    --max-card-files)
      MAX_CARD_FILES="$2"
      shift 2
      ;;
    --line-nos)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        LINE_NOS+=("$1")
        shift
      done
      ;;
    --all-regions)
      ALL_REGIONS=1
      shift
      ;;
    --no-progress)
      NO_PROGRESS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"

ARGS=(
  run --no-sync python data_processing/line_bus/prepare_line_od.py
  --interval-minutes "$GRANULARITY"
  --min-total-flow "$MIN_TOTAL_FLOW"
  --min-stops "$MIN_STOPS"
  --poi-top-categories "$POI_TOP_CATEGORIES"
  --use-card-line-dirs
)
if [[ "$ALL_REGIONS" != "1" ]]; then
  ARGS+=(--regions "$REGION")
fi
if [[ "$MAX_LINES" != "0" ]]; then
  ARGS+=(--max-lines "$MAX_LINES")
fi
if [[ "$MAX_CARD_FILES" != "0" ]]; then
  ARGS+=(--max-card-files "$MAX_CARD_FILES")
fi
if [[ ${#LINE_NOS[@]} -gt 0 ]]; then
  ARGS+=(--line-nos "${LINE_NOS[@]}")
fi
if [[ "$NO_PROGRESS" == "1" ]]; then
  ARGS+=(--no-progress)
fi

uv "${ARGS[@]}"
