#!/bin/bash
#SBATCH --job-name=mjx_ars
#SBATCH --account=ealloc_ati-1-neur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mjx_ars_%j.out
#SBATCH --error=logs/mjx_ars_%j.err

module purge
module load python/3.11.14

source ~/venvs/mjx/bin/activate

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd ~/mujoco_ars_project

python -u run_deliverable4_hpc.py \
  --run-mode strong_stable \
  --cem-mode fairer_slow \
  --train-seconds 6 \
  --video-seconds 6 \
  --output-dir results_mjx_stable
