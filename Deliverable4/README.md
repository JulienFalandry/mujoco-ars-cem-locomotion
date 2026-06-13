\# MuJoCo MJX Locomotion Optimisation using CEM and ARS



This repository contains the code and results for the final project of the Optimisation course.

The goal is to compare gradient-free optimisation methods, mainly the Cross-Entropy Method (CEM) and Augmented Random Search (ARS), on a MuJoCo MJX locomotion task.



\## Project overview



The project studies how CEM and ARS behave on a high-dimensional continuous-control locomotion problem.

CEM is used as a baseline attempt, while ARS is used as the main scalable optimiser for the walking policy.



The implementation uses:



\* MuJoCo / MJX for physics simulation

\* JAX for batched GPU-accelerated rollouts

\* CEM as a baseline optimiser

\* ARS with state normalisation for locomotion optimisation

\* Slurm scripts for running the experiments on the University of Tartu HPC GPU cluster



\## Repository structure



```text

.

├── run\_deliverable4\_hpc.py      # Main training and evaluation script

├── submit\_mjx\_gpu.slurm         # Slurm script for running the full experiment on HPC

├── test\_jax\_gpu.slurm           # Small script to verify JAX GPU access

├── setup\_env.sh                 # Environment setup helper for HPC

├── requirements\_hpc.txt         # Python dependencies

├── README\_HPC.md                # Additional HPC notes

└── results\_mjx/

&#x20;   ├── learning\_curves.png      # CEM vs ARS learning curve

&#x20;   ├── metrics.json             # Final numerical metrics

&#x20;   ├── best\_ars\_policy.npz      # Saved best ARS policy

&#x20;   └── best\_ars\_walker2d\_policy\_over\_6.0s\_\_strong\_balanced.mp4

```



\## Methods



\### Cross-Entropy Method



CEM samples a population of candidate policies from a Gaussian distribution, evaluates them in parallel, selects the best-performing elite policies, and updates the distribution mean and standard deviation.



In this project, CEM is used as a baseline attempt on the locomotion task. It improves compared with random policies, but it remains much less scalable than ARS for the high-dimensional policy search problem.



\### Augmented Random Search



ARS updates a linear policy by sampling random perturbation directions, evaluating positive and negative perturbations, and moving the policy parameters in the direction of the best-performing perturbations.



The ARS implementation includes state normalisation and batched MJX rollouts, allowing many candidate policies to be evaluated in parallel on GPU.



\## Final result



The best ARS configuration was `strong\_balanced`.



Main diagnostic result over a 6-second horizon:



```text

Best ARS reward: 9349.90

Best iteration: 236

Distance travelled: 10.82 m

Mean forward velocity: 1.81 m/s

First fallen step: 600 / 600

```



The result shows that ARS successfully learned a forward locomotion gait, while CEM remained much less effective on the same high-dimensional control problem.



\## Sample efficiency



Approximate number of physics steps:



```text

CEM: 34,560,000 steps

ARS: 277,200,000 steps

```



Although ARS used more total environment steps in the final strong run, it produced a much better locomotion policy and scaled better to the high-dimensional continuous control setting.



\## How to run on the HPC



First, create and activate the Python environment:



```bash

bash setup\_env.sh

source \~/venvs/mjx/bin/activate

```



Then submit the GPU job:



```bash

sbatch submit\_mjx\_gpu.slurm

```



To monitor the job:



```bash

squeue -u $USER

tail -f logs/mjx\_ars\_\*.out

```



The results are saved in:



```text

results\_mjx/

```



\## Requirements



The main dependencies are listed in `requirements\_hpc.txt`.



Main packages:



```text

jax

mujoco

mujoco-mjx

numpy

matplotlib

mediapy

imageio-ffmpeg

```



\## Notes



The experiment was run on the University of Tartu HPC GPU cluster using a Tesla V100 GPU.

The video rendering requires `ffmpeg`; on the HPC, this was provided through the Python virtual environment using `imageio-ffmpeg`.



