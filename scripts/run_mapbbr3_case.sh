#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec gidind-ai infer \
  --input "$ROOT/examples/real_cases/mapbbr3.csv" \
  --output-dir "${GIDIND_AI_OUTPUT_DIR:-$ROOT/reviewer_outputs/mapbbr3}" \
  --case-profile mapbbr3 \
  --release-model standard \
  "$@"
