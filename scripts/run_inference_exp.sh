#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=4:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N terc_inference
#$ -t 1-50

set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_inference.txt"

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
mkdir -p results/individual
mkdir -p models_trained
mkdir -p logs/inference_exp

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_inference.txt:
#   <env> <state_type> <seed>
env_name=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
state_type=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $3}')

if [[ -z "$env_name" || -z "$state_type" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running TERC inference experiment: env=$env_name, state_type=$state_type, seed=$seed"

# ---------------------------------------------------------------------
# 5.  Check GPU availability before running
# ---------------------------------------------------------------------
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
else
  echo "nvidia-smi not available"
fi

# ---------------------------------------------------------------------
# 6.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting experiment..."
python3.9 -u run_single.py \
    --env "$env_name" \
    --state_type "$state_type" \
    --seed "$seed" \
    --device cuda \
    --output_dir results/individual \
    --model_dir models_trained \
    --n_warmup 1000 \
    --n_measure 10000

date
echo "Experiment completed: $env_name $state_type seed=$seed"
