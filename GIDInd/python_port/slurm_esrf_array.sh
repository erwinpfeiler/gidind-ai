#!/bin/bash
#SBATCH --partition=nice-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --array=0-5
#SBATCH --job-name=gidind-array
#SBATCH --output=gidind-array-%A_%a.out
#SBATCH --error=gidind-array-%A_%a.err
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: sbatch $0 RUN_DIRECTORY" >&2; exit 2; fi
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a SLURM array}"
cd "${GIDIND_PROJECT_ROOT:-$(pwd)}"
python_bin="$PWD/GIDInd/python_port/.venv/bin/python"
workers=${SLURM_CPUS_PER_TASK:?Missing CPU allocation}
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Array %s task %s; host %s; workers %s; output root %s\n' \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$workers" "$1"
exec srun --ntasks=1 --cpus-per-task="$workers" "$python_bin" -m gidind_reference.slurm_array \
  run --run-dir "$1" --task-id "$SLURM_ARRAY_TASK_ID"
