#!/usr/bin/env bash
set -euo pipefail

REGION="guangdianyuan"
GRANULARITY="60"
DRY_RUN=0
SKIP_COMPARE=0

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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-compare)
      SKIP_COMPARE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CONFIG="configs/line_bus/${REGION}_${GRANULARITY}_base.yaml"
ARGS=(--suite configs/line_bus/suite_main.yaml --base-config "$BASE_CONFIG" --only od_llm)
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$SKIP_COMPARE" == "1" ]]; then
  ARGS+=(--skip-compare)
fi

bash "$PROJECT_ROOT/scripts/run_suite.sh" "${ARGS[@]}"
