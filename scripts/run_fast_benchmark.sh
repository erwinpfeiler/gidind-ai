#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec gidind-ai benchmark \
  --output-dir "${GIDIND_AI_OUTPUT_DIR:-$ROOT/reviewer_outputs/fast_benchmark}" \
  "$@"
