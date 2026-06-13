# Deliverable 4 — Locomotion Optimisation with ARS and CEM

This folder contains the code and results for Deliverable 4 of the Optimisation class project: locomotion optimisation using the Cross-Entropy Method (CEM) and Augmented Random Search (ARS) with MuJoCo MJX and JAX.

The original assignment asks for ARS on Humanoid locomotion. In this implementation, the final locomotion experiment is performed on a Walker2d MuJoCo model, which is used as a simpler bipedal locomotion proxy. The goal remains the same: compare the scalability of CEM and ARS on a high-dimensional continuous-control locomotion task.

## Folder structure

```text
Deliverable4/
├── README.md
├── requirements.txt
├── run_deliverable4_hpc.py
└── results_mjx/
    ├── learning_curves.png
    ├── metrics.json
    ├── best_ars_policy.npz
    └── best_ars_walker2d_policy_over_6.0s__strong_balanced.mp4
```

## Main script

The main experiment is implemented in:

```text
run_deliverable4_hpc.py
```

This script performs:

* Walker2d MuJoCo model creation
* Transfer of the model to MJX
* Batched rollouts using `jax.vmap`
* CEM baseline optimization
* ARS optimisation with state normalisation
* Learning curve generation
* Policy diagnostics
* Video rendering of the optimised gait
* Saving of final metrics and policy parameters

## Methods

### Cross-Entropy Method

CEM samples a population of candidate policies from a Gaussian distribution. At each iteration, the best-performing elite policies are selected and used to update the sampling distribution.

In this project, CEM is used as a baseline attempt on the locomotion task. It improves compared with random policies but remains much less effective than ARS for the high-dimensional continuous-control problem.

### Augmented Random Search

ARS evaluates random perturbations of a linear policy. For each direction, the positive and negative perturbations are tested, and the policy is updated using the best-performing directions.

The implemented ARS version uses state normalisation and evaluates perturbations in parallel using MJX and JAX. This makes it more scalable than CEM for locomotion.

## Hyperparameter tuning

The ARS implementation allows tuning of the main parameters requested in the assignment:

```text
α  = step size
N  = number of perturbation directions
b  = number of top-performing directions used for the update
ν  = exploration noise
```

The final run used the latest tuned reward and ARS configuration available in the repository code. Although the configuration label still appears as `strong_balanced` / `strong_stable` in some generated files, the code contains the final reward-shaping updates.

## Final results

The final ARS policy produced a stable forward gait over the 6-second evaluation horizon.

Main final metrics:

```text
Distance travelled: approximately 4.58 m
Mean forward velocity: approximately 0.76 m/s
First fallen step: 600 / 600
```

This indicates that the final ARS policy remains upright for the full evaluation horizon and moves forward with a visible walking gait.

The CEM baseline remained significantly weaker, confirming that CEM is less suitable for this higher-dimensional locomotion task.

## Sample efficiency

The experiment also reports the approximate number of physics steps used by each method.

The final results are stored in:

```text
results_mjx/metrics.json
```

The learning curve comparison is stored in:

```text
results_mjx/learning_curves.png
```

The final rendered policy video is stored in:

```text
results_mjx/best_ars_walker2d_policy_over_6.0s__strong_balanced.mp4
```

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment:

```bash
python run_deliverable4_hpc.py \
  --run-mode strong \
  --cem-mode fairer_slow \
  --train-seconds 6 \
  --video-seconds 6 \
  --output-dir results_mjx
```

For live output in an HPC or Slurm environment, use:

```bash
python -u run_deliverable4_hpc.py \
  --run-mode strong \
  --cem-mode fairer_slow \
  --train-seconds 6 \
  --video-seconds 6 \
  --output-dir results_mjx
```

## Notes

The final experiment was run on a GPU using JAX/MJX batched rollouts. This greatly accelerated the evaluation of candidate policies compared with sequential CPU rollouts.

The final report discusses:

* why ARS scales better than CEM,
* the effect of JAX/MJX parallelisation,
* the sample-efficiency comparison,
* the limitations of using Walker2d instead of the full Humanoid model,
* and the final locomotion performance.
