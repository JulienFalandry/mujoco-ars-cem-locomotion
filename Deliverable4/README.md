# Deliverable 4 — Locomotion Optimization with ARS and CEM

This folder contains the code and results for Deliverable 4 of the Optimization class project: locomotion optimization using Augmented Random Search (ARS) and the Cross Entropy Method (CEM) with MuJoCo MJX and JAX.

The original assignment focuses on Humanoid locomotion. In this implementation, the final experiment is performed on a custom Walker2d MuJoCo model, used as a simpler bipedal locomotion proxy. The objective remains the same: compare how CEM and ARS scale on a high-dimensional continuous-control locomotion task.

## Folder structure

```text
Deliverable4/
├── README.md
├── requirements.txt
├── run_deliverable4_hpc.py
├── Deliverable4_Final_Report.pdf
└── results_mjx/
    ├── metrics.json
    ├── learning_curves.png
    ├── best_ars_policy.npz
    ├── best_ars_walker2d_policy_over_6.0s__strong_stable.mp4
    └── best_cem_walker2d_policy_over_6.0s__fairer_long.mp4
```

## Main script

The main script is:

```text
run_deliverable4_hpc.py
```

It performs the full locomotion experiment:

* creates the Walker2d MuJoCo model;
* transfers the model to MJX;
* runs batched rollouts with JAX;
* trains a policy with CEM;
* trains a policy with ARS;
* saves metrics and learning curves;
* renders the best ARS policy video;
* optionally renders the best CEM policy video.

## Methods

### Cross Entropy Method

CEM samples a population of candidate policies from a Gaussian distribution. Each candidate is evaluated over the rollout horizon, then the best candidates are selected as elites. The Gaussian mean and standard deviation are updated from these elites.

For the final version, CEM was extended using the `fairer_long` mode. This longer CEM baseline was added to make the comparison more meaningful than the previous shorter CEM run.

### Augmented Random Search

ARS optimizes a linear policy by sampling random perturbation directions around the current parameters. For each direction, both the positive and negative perturbations are evaluated. The update is computed from the best-performing directions.

The final ARS run uses the `strong_stable` mode. State normalization is included to improve training stability.

## Final results

The final experiment was run on the GPU backend with a 6-second rollout horizon and 6-second video rendering horizon.

| Method |            Mode | Physics steps | Best reward |
| ------ | --------------: | ------------: | ----------: |
| CEM    |   `fairer_long` |    92,160,000 |     1698.54 |
| ARS    | `strong_stable` |   323,400,000 |     5780.55 |

The final ARS policy achieved the following diagnostic performance:

| Metric                 |     Value |
| ---------------------- | --------: |
| Distance travelled     |    6.94 m |
| Mean forward velocity  |  1.16 m/s |
| Minimum torso height   |    0.94 m |
| Maximum absolute pitch | 0.676 rad |
| First fall step        | 600 / 600 |

The extended CEM run improved the baseline compared with the previous shorter CEM experiment, but ARS still achieved a much stronger final policy. The ARS policy produced stable forward locomotion over the full 6-second rollout, while the CEM policy remained visibly weaker in the rendered video.

## Learning curves

The file:

```text
results_mjx/learning_curves.png
```

shows the training curves for both methods.

The curve confirms that ARS reaches a significantly higher final reward than CEM on this locomotion task. CEM improves with the longer `fairer_long` run, but it remains less effective for this high-dimensional continuous-control problem.

## Output videos

The final ARS video is:

```text
results_mjx/best_ars_walker2d_policy_over_6.0s__strong_stable.mp4
```

The final CEM video is:

```text
results_mjx/best_cem_walker2d_policy_over_6.0s__fairer_long.mp4
```

The ARS video is the main optimized locomotion result. The CEM video is included as an additional comparison to show the weaker behavior of the CEM policy.

## How to reproduce the final run

The final experiment can be reproduced with:

```bash
python -u run_deliverable4_hpc.py \
  --run-mode strong_stable \
  --cem-mode fairer_long \
  --train-seconds 6 \
  --video-seconds 6 \
  --output-dir results_mjx \
  --render-cem
```

The `--render-cem` flag is required to generate the CEM video. Without it, the script only renders the ARS policy video.

## Notes

* The implementation uses Walker2d as a bipedal locomotion proxy instead of the full Humanoid environment.
* The final comparison is not perfectly equal-budget, because ARS uses more physics steps than CEM.
* The goal of the final experiment is to show that ARS scales better than CEM on this type of high-dimensional locomotion problem.
* The `.slurm` launcher file is kept local because it depends on the HPC account, partition, and GPU configuration.
