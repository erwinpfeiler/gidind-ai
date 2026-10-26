#!/bin/bash
#SBATCH --partition=nice-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=gidind-collect
#SBATCH --output=gidind-collect-%j.out
#SBATCH --error=gidind-collect-%j.err
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: sbatch $0 RUN_DIRECTORY" >&2; exit 2; fi
: "${SLURM_JOB_ID:?Submit collection with sbatch}"
cd "${GIDIND_PROJECT_ROOT:-$(pwd)}"
export PYTHONUNBUFFERED=1
exec srun --ntasks=1 --cpus-per-task=1 "$PWD/GIDInd/python_port/.venv/bin/python" \
  -m gidind_reference.slurm_array collect --run-dir "$1"
