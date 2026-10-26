#!/bin/bash
# Run on the ESRF login node; prepares inputs/settings before submitting anything.
set -euo pipefail
mode=${1:-}
if [[ $# -lt 1 || ! "$mode" =~ ^(smoke|full|ladder)$ ]]; then
    echo "Usage: bash $0 smoke|full|ladder [NEW_RUN_DIRECTORY] [--algorithm corrected|upstream] [--afterok JOBID]" >&2; exit 2
fi
shift
requested_run_dir=''
algorithm_override=()
dependency=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo '--afterok requires a numeric job ID' >&2; exit 2
            fi
            dependency=(--dependency="afterok:$2" --kill-on-invalid-dep=yes); shift 2 ;;
        --algorithm)
            if [[ $# -lt 2 || ! "$2" =~ ^(corrected|upstream)$ ]]; then
                echo '--algorithm requires corrected or upstream' >&2; exit 2
            fi
            algorithm_override=(--algorithm "$2"); shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if [[ -n "$requested_run_dir" ]]; then echo 'Only one output directory is allowed' >&2; exit 2; fi
            requested_run_dir=$1; shift ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this submission helper on the login node, outside an existing SLURM allocation." >&2; exit 2
fi
cd "${GIDIND_PROJECT_ROOT:-$(pwd)}"
port_root="$PWD/GIDInd/python_port"
python_bin="$port_root/.venv/bin/python"
command -v sbatch >/dev/null || { echo "sbatch is unavailable; use the ESRF login node." >&2; exit 1; }
bash "$port_root/slurm_esrf.sh" preflight
run_dir=${requested_run_dir:-$port_root/benchmark_runs/esrf_array_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$}
run_dir=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$run_dir")
budget=7200
allocation=02:30:00
if [[ "$mode" == smoke ]]; then budget=120; allocation=00:10:00; fi
budget_override=()
if [[ "$mode" == ladder ]]; then budget_override=(--budgets-seconds 120 600 3600 7200); allocation=04:00:00; fi
"$python_bin" -m gidind_reference.slurm_array prepare \
  --input-dir "$PWD/gidind_peak_lists_all" --run-dir "$run_dir" \
  --config "$port_root/configs/revision_2h_32cpu.json" --workers 32 --time-limit-seconds "$budget" ${algorithm_override[@]+"${algorithm_override[@]}"} ${budget_override[@]+"${budget_override[@]}"}
"$python_bin" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
plan=json.loads((Path(sys.argv[1])/'array_plan.json').read_text())
if len(plan['tasks'])!=6 or len(plan['entries'])!=8:
    raise SystemExit('This ESRF launcher expects six unique tasks and eight entries; no jobs submitted.')
PY
array_id=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --time="$allocation" "$port_root/slurm_esrf_array.sh" "$run_dir")
array_id=${array_id%%;*}
printf 'Submitted array: %s\nOutput root: %s\n' "$array_id" "$run_dir"
printf '%s\n' "$array_id" > "$run_dir/array_job_id.txt"
if collector_id=$(sbatch --parsable --dependency="afterany:$array_id" "$port_root/slurm_esrf_collect.sh" "$run_dir"); then
    collector_id=${collector_id%%;*}
    printf '%s\n' "$collector_id" > "$run_dir/collector_job_id.txt"
    printf 'Collector job: %s (starts after every array task ends)\n' "$collector_id"
else
    echo "The array IS submitted, but collector submission failed. After it ends, run:" >&2
    printf 'sbatch %q %q\n' "$port_root/slurm_esrf_collect.sh" "$run_dir" >&2
    exit 1
fi
printf 'Monitor: squeue -r -j %s\nLogs: gidind-array-%s_TASKID.out / .err\n' "$array_id" "$array_id"
printf 'Keep the checkout/environment unchanged until array and collector finish.\n'
