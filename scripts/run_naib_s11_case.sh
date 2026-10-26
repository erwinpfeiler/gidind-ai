#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec gidind-ai infer \
  --input "$ROOT/examples/real_cases/naib_s11_without_specular.csv" \
  --output-dir "${GIDIND_AI_OUTPUT_DIR:-$ROOT/reviewer_outputs/naib_s11}" \
  --case-profile naib_s11 \
  "$@"
