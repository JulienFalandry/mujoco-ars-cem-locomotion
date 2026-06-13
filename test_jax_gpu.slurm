#!/bin/bash

#SBATCH --job-name=test_jax_gpu

#SBATCH --account=ealloc_ati-1-neur

#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --time=00:10:00

#SBATCH --output=logs/test_jax_gpu_%j.out

#SBATCH --error=logs/test_jax_gpu_%j.err

module purge
module load python/3.11.14

source ~/venvs/mjx/bin/activate

echo "Node: $(hostname)"
nvidia-smi

python - <<'PY'
import sys
print("Python:", sys.version)

import jax
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

import mujoco
from mujoco import mjx
print("MuJoCo:", mujoco.__version__)
print("MJX import OK")
PY
