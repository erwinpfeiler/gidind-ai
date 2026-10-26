#!/bin/bash
# Submit smoke + a full array that runs only after technical smoke success.
set -euo pipefail
if [[ $# -gt 1 ]]; then echo "Usage: bash $0 [NEW_PARENT_DIRECTORY]" >&2; exit 2; fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then echo 'Run on the login node, outside an allocation.' >&2; exit 2; fi
cd "${GIDIND_PROJECT_ROOT:-$(pwd)}"
port_root="$PWD/GIDInd/python_port"
parent=${1:-$port_root/benchmark_runs/revision_2h_32cpu_$(date -u +%Y%m%dT%H%M%SZ)_$$}
if [[ -e "$parent" ]]; then echo "Refusing to reuse directory: $parent" >&2; exit 2; fi
mkdir -p "$parent"
parent=$(cd "$parent" && pwd)
bash "$port_root/submit_esrf_array.sh" smoke "$parent/smoke" --algorithm corrected
gate=$(cat "$parent/smoke/collector_job_id.txt")
bash "$port_root/submit_esrf_array.sh" full "$parent/full" --algorithm corrected --afterok "$gate"
printf '\nRevision output: %s\n' "$parent"
printf 'Full searches: 32 CPUs/case, 7200 seconds/case; queued behind smoke collector %s.\n' "$gate"
printf 'Smoke checks technical completion, never reference correctness.\n'
printf 'A failed smoke collector prevents the full array from starting.\n'
printf 'Keep checkout, inputs, settings and environment unchanged until all jobs finish.\n'
