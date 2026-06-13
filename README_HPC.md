# MJX Deliverable 4 on UT HPC

Files:
- `run_deliverable4_hpc.py`: non-interactive script version of the notebook.
- `requirements_hpc.txt`: Python packages.
- `setup_env.sh`: creates a Python venv and Jupyter kernel.
- `submit_mjx_gpu.slurm`: submits a GPU job.

Quick commands:

```bash
cd /path/to/project
bash setup_env.sh
mkdir -p logs results_mjx
sbatch submit_mjx_gpu.slurm
squeue -u $USER
```

Fast test before the real job:

```bash
source ~/venvs/mjx/bin/activate
python run_deliverable4_hpc.py --run-mode smoke --cem-mode smoke --video-seconds 4 --output-dir test_results
```

Outputs are saved in `results_mjx/`: learning curve PNG, video MP4, `metrics.json`, and `best_ars_policy.npz`.
