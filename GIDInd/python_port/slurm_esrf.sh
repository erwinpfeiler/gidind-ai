#!/bin/bash
#SBATCH --partition=nice-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --job-name=gidind-cpu
#SBATCH --output=gidind-%j.out
#SBATCH --error=gidind-%j.err

set -euo pipefail

mode=${1:-}
if [[ $# -ne 1 || ! "$mode" =~ ^(preflight|smoke|full)$ ]]; then
    echo "Usage: bash $0 preflight | sbatch [options] $0 smoke|full" >&2
    exit 2
fi
project_root=${GIDIND_PROJECT_ROOT:-$(pwd)}
cd "$project_root"
port_root="$PWD/GIDInd/python_port"
python_bin="$port_root/.venv/bin/python"
workers=${SLURM_CPUS_PER_TASK:-8}
if [[ ! "$workers" =~ ^[0-9]+$ || ! "$workers" =~ [1-9] ]]; then
    echo "SLURM_CPUS_PER_TASK must be a positive integer." >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing $python_bin. Run GIDInd/python_port/setup.sh on the login node first." >&2
    exit 1
fi
if [[ "$mode" != preflight && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit smoke/full with sbatch; do not run indexing on the login node." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Login-node preflight and compute-node startup use exactly the same checks.
# No downloads, pip installations, or Hugging Face tokens are needed in the job.
"$python_bin" - <<'PY'
import sys
from pathlib import Path
import json
import sqlite3
if sys.version_info < (3,12):
    raise SystemExit('The pinned environment requires Python 3.12 or newer. Recreate the dedicated port environment.')
import gidind_reference
import numpy
import scipy
import openpyxl
sys.path.insert(0, str(Path('scripts').resolve()))
from download_real_case_peak_lists import MANIFEST_SHA256, verify
root = Path('gidind_peak_lists_all')
manifest_path = root/'manifest.json'
if not manifest_path.is_file():
    raise SystemExit('Missing peak lists. Run scripts/download_real_case_peak_lists.py --include-originals --output-dir gidind_peak_lists_all on the login node.')
payload = manifest_path.read_bytes()
verify(payload, MANIFEST_SHA256, 'manifest.json')
manifest = json.loads(payload)
files = 0
for case in manifest['cases']:
    derived = case['without_specular']
    for name, digest in ((case['filename'], derived['source_sha256']),
                         (derived['filename'], derived['sha256'])):
        if Path(name).name != name:
            raise SystemExit(f'Invalid manifest filename: {name}')
        path = root/name
        if not path.is_file():
            raise SystemExit(f'Missing {path}; download all eight CSVs on the login node.')
        verify(path.read_bytes(), digest, name)
        files += 1
print(f'Preflight OK: Python {sys.version.split()[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, openpyxl {openpyxl.__version__}, SQLite {sqlite3.sqlite_version}; {files} verified CSVs.', flush=True)
PY

if [[ "$mode" == preflight ]]; then
    echo "Ready to submit the smoke test. No indexing was run."
    exit 0
fi

output_dir="$port_root/benchmark_runs/esrf_${mode}_${SLURM_JOB_ID}"
if [[ -e "$output_dir" ]]; then
    echo "Refusing to reuse $output_dir. Submit a fresh job for a fresh directory." >&2
    exit 1
fi
printf 'Mode: %s | job: %s | host: %s | workers: %s\n' "$mode" "$SLURM_JOB_ID" "$(hostname)" "$workers"
printf 'Started (UTC): %s\n' "$(date -u +%FT%TZ)"
printf 'Source commit: %s\n' "$(git rev-parse HEAD)"
printf 'Output directory: %s\n' "$output_dir"

args=(benchmark --input-dir "$PWD/gidind_peak_lists_all"
      --config "$port_root/configs/benchmark_settings.json"
      --workers "$workers" --output-dir "$output_dir")
if [[ "$mode" == smoke ]]; then
    args+=(--time-limit-seconds 30)
    echo "Smoke test: 30 seconds per unique case; partial searches are expected."
else
    echo "Benchmark: two-hour budget per unique case, six unique cases sequentially."
fi
exec srun --ntasks=1 --cpus-per-task="$workers" "$port_root/run.sh" ${args[@]+"${args[@]}"}
