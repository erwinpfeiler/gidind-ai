#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec gidind-ai infer \
  --input "$ROOT/examples/real_cases/pq_without_specular.csv" \
  --output-dir "${GIDIND_AI_OUTPUT_DIR:-$ROOT/reviewer_outputs/pq}" \
  --case-profile pq \
  "$@"
