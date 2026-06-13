#!/bin/bash
set -euo pipefail

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
