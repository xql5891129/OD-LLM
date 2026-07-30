#!/usr/bin/env bash
set -euo pipefail

REGION="guangdianyuan"
GRANULARITY="60"
DRY_RUN=0
CONTINUE_ON_ERROR=0
SKIP_COMPARE=0
ONLY=(ha lstm transformer odmixer odcrn)

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
    --only)
      ONLY=()
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
ARGS=(--suite configs/line_bus/suite_baselines.yaml --base-config "$BASE_CONFIG")
if [[ ${#ONLY[@]} -gt 0 ]]; then
  ARGS+=(--only "${ONLY[@]}")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi
if [[ "$SKIP_COMPARE" == "1" ]]; then
  ARGS+=(--skip-compare)
fi

bash "$PROJECT_ROOT/scripts/run_suite.sh" "${ARGS[@]}"
