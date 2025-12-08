#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=4:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N terc_condition1
#$ -t 1-65

set -euo pipefail

hostname
date

# ---------------------------------------------------------------------
# 0.  Determine script directory and change to repo root
# ---------------------------------------------------------------------
# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up two levels to repo root
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"
echo "Working directory: $(pwd)"

number=$SGE_TASK_ID
paramfile="condition1_validation/scripts/jobs_condition1.txt"

# Check if paramfile exists
if [[ ! -f "$paramfile" ]]; then
  echo "ERROR: Parameter file not found: $paramfile" >&2
  echo "Current directory: $(pwd)" >&2
  echo "Contents: $(ls -la)" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
  source /share/apps/source_files/cuda/cuda-11.8.source || true
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  if [[ -f /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate ]]; then
    source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
  fi
fi

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p condition1_validation/results/individual
mkdir -p condition1_validation/figures
mkdir -p trajectories
mkdir -p logs/condition1_exp

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line: <env> <seed>
env_name=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

if [[ -z "$env_name" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running Condition 1 validation: env=$env_name, seed=$seed"

# ---------------------------------------------------------------------
# 5.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting validation..."
python3.9 -u condition1_validation/run_single_validation.py \
    --env "$env_name" \
    --seed "$seed" \
    --n_samples 10000 \
    --k_neighbors 5 \
    --data_dir trajectories \
    --model_dir models_trained \
    --output_dir condition1_validation/results/individual \
    --save_trajectories

date
echo "Validation completed: $env_name seed=$seed"
