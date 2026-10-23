#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec gidind-ai infer \
  --input "$ROOT/examples/real_cases/cu_bod_mof.csv" \
  --output-dir "${GIDIND_AI_OUTPUT_DIR:-$ROOT/reviewer_outputs/cu_bod_mof}" \
  --case-profile cu_bod_mof \
  --release-model standard \
  "$@"
