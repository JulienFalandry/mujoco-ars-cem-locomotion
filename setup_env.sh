#!/bin/bash
set -euo pipefail

# Run once in an OnDemand terminal or SSH session.
# If the cluster provides a Python/conda module, load it first. Examples:
# module avail python
# module load any/python/3.11.6

python -m venv ~/venvs/mjx
source ~/venvs/mjx/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_hpc.txt
python -m ipykernel install --user --name=mjx --display-name="Python (MJX)"

python - <<'PY'
import jax
print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())
PY
