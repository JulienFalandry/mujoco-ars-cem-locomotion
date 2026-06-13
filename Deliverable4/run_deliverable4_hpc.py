# Headless rendering for HPC/Slurm jobs. Must be set before importing mujoco.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json as _json

parser = argparse.ArgumentParser(description="Run Deliverable 4 Walker2d ARS/CEM training on HPC")
parser.add_argument(
    "--run-mode",
    default="strong",
    choices=["smoke", "colab_balanced", "colab_sweep", "strong", "strong_stable", "strong_middle"],
    help="ARS tuning mode. Use smoke for a quick test, strong/strong_stable for HPC.")
parser.add_argument("--cem-mode", default="fairer_slow", choices=["smoke", "extended_colab", "fairer_slow"],
                    help="CEM baseline mode. fairer_slow is suitable for HPC.")
parser.add_argument("--train-seconds", type=float, default=6.0, help="Training rollout duration in seconds.")
parser.add_argument("--video-seconds", type=float, default=12.0, help="Rendered video duration in seconds.")
parser.add_argument("--output-dir", default="results_mjx", help="Directory where plots, policy, metrics, and videos are saved.")
parser.add_argument("--render-cem", action="store_true", help="Also render the best CEM policy.")
ARGS = parser.parse_args()
os.makedirs(ARGS.output_dir, exist_ok=True)

# Import numerical, plotting, JAX, MJX, and video libraries.
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import lax, random

import mujoco
from mujoco import mjx
import mediapy as media

# Print device information for the final report.
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# %%
# Define the Walker2d locomotion model from the provided notebook.
# Contact fix: use contype="1" and conaffinity="0" for body geoms.
# This lets body geoms contact the floor, whose conaffinity is 1, without enabling self-collision.
WALKER2D_XML = r"""
<mujoco model="walker2d">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint limited="true" damping=".05" armature=".1"/>
    <geom contype="1" conaffinity="0" condim="3" density="1000" friction="0.7 0.1 0.1"/>
  </default>
  <option timestep="0.002" integrator="Euler"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2=".8 .8 .8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba=".8 .9 .8 1" size="40 40 40" type="plane" material="MatPlane"/>

    <body name="torso" pos="0 0 1.25">
      <camera name="track" mode="trackcom" pos="0 -5 0.8" xyaxes="1 0 0 0 0 1" fovy="45"/>
      <joint armature="0" damping="0" limited="false" name="rootx" type="slide" axis="1 0 0"/>
      <joint armature="0" damping="0" limited="false" name="rootz" type="slide" axis="0 0 1" ref="1.25"/>
      <joint armature="0" damping="0" limited="false" name="rooty" type="hinge" axis="0 1 0"/>
      <geom friction="0.9" fromto="0 0 0.07 0 0 0.45" name="torso_geom" size="0.05" type="capsule"/>

      <body name="thigh" pos="0 0 0.07">
        <joint axis="0 -1 0" name="thigh_joint" range="-2.6 0.05" type="hinge"/>
        <geom friction="0.9" fromto="0 0 0 0 0 -0.45" name="thigh_geom" size="0.05" type="capsule"/>
        <body name="leg" pos="0 0 -0.45">
          <joint axis="0 -1 0" name="leg_joint" range="-2.6 -0.1" type="hinge"/>
          <geom friction="0.9" fromto="0 0 0 0 0 -0.5" name="leg_geom" size="0.04" type="capsule"/>
          <body name="foot" pos="0 0 -0.5">
            <joint axis="0 -1 0" name="foot_joint" range="-0.8 0.8" type="hinge"/>
            <geom friction="0.9" fromto="-0.2 0 0 0.2 0 0" name="foot_geom" size="0.06" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="thigh_left" pos="0 0 0.07">
        <joint axis="0 -1 0" name="thigh_left_joint" range="-2.6 0.05" type="hinge"/>
        <geom friction="0.9" fromto="0 0 0 0 0 -0.45" name="thigh_left_geom" size="0.05" type="capsule"/>
        <body name="leg_left" pos="0 0 -0.45">
          <joint axis="0 -1 0" name="leg_left_joint" range="-2.6 -0.1" type="hinge"/>
          <geom friction="0.9" fromto="0 0 0 0 0 -0.5" name="leg_left_geom" size="0.04" type="capsule"/>
          <body name="foot_left" pos="0 0 -0.5">
            <joint axis="0 -1 0" name="foot_left_joint" range="-0.8 0.8" type="hinge"/>
            <geom friction="0.9" fromto="-0.2 0 0 0.2 0 0" name="foot_left_geom" size="0.06" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="thigh_joint"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="leg_joint"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="foot_joint"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="thigh_left_joint"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="leg_left_joint"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="100" joint="foot_left_joint"/>
  </actuator>
</mujoco>
"""

# Build the CPU MuJoCo model and transfer it to MJX.
model = mujoco.MjModel.from_xml_string(WALKER2D_XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mx_model = mjx.put_model(model)
mx_data = mjx.make_data(mx_model)

# One control step holds the action for several fine physics steps.
N_SUBSTEPS = 5
DT_CTRL = model.opt.timestep * N_SUBSTEPS

print(f"Walker2d loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"Control dt: {DT_CTRL:.4f}s")

# Sanity check: with contacts enabled, a passive walker should contact the floor instead of falling through it.
check_data = mujoco.MjData(model)
for _ in range(400):
    mujoco.mj_step(model, check_data)
print(f"Contact sanity check: torso height={check_data.qpos[1]:.3f}, active contacts={check_data.ncon}")

# %%
# Define rollout and policy sizes.
# Keep training horizon moderate for Colab. The video length is controlled separately later.
TRAIN_SECONDS = ARGS.train_seconds
HORIZON = int(TRAIN_SECONDS / DT_CTRL)

NU = model.nu
NQ = model.nq
NV = model.nv

# Exclude absolute x position from the policy observation.
STATE_DIM = (NQ - 1) + NV
PARAM_DIM = NU * STATE_DIM + NU

# Reward shaping hyperparameters for walking forward without falling.
# Balanced version: the previous stable reward survived, but did not walk.
# This version makes standing still unattractive while still rewarding survival.
TARGET_VEL = 1.05
H_MIN = 0.80
H_TARGET = 1.25
ANGLE_MAX = 0.90


# Extract the policy observation: all qpos except absolute x, plus all qvel.
def walker_obs(d):
    return jnp.concatenate([d.qpos[1:], d.qvel])


# Convert one flat parameter vector into policy weights and bias.
def unpack_policy(params):
    weights = params[: NU * STATE_DIM].reshape(NU, STATE_DIM)
    bias = params[NU * STATE_DIM :]
    return weights, bias


# Compute bounded motor commands from the normalized observation.
def policy_action(params, obs, obs_mean, obs_std):
    weights, bias = unpack_policy(params)
    obs_norm = (obs - obs_mean) / obs_std
    return jnp.tanh(weights @ obs_norm + bias)


# Reward one control step: go forward, stay upright, and use moderate/smooth control.
def walker_step_reward(d0, d1, action, prev_action):
    vx = (d1.qpos[0] - d0.qpos[0]) / DT_CTRL
    height = d1.qpos[1]
    pitch = d1.qpos[2]
    pitch_rate = d1.qvel[2]

    alive = ((height > H_MIN) & (jnp.abs(pitch) < ANGLE_MAX)).astype(jnp.float32)

    # Strong enough forward reward so the policy does not learn to simply stand.
    forward_reward = 7 * vx
    velocity_tracking = -1.0 * (vx - TARGET_VEL) ** 2

    # Small alive reward only. If this is too large, standing becomes attractive.
    healthy_reward = 0.25 * alive

    # Explicitly punish no-progress policies.
    no_progress_penalty = 3 * (vx < 0.12).astype(jnp.float32)

    # Mild stability penalties. Too much stability penalty kills walking.
    height_cost = 0.6 * (height - H_TARGET) ** 2
    posture_cost = 0.8 * pitch**2
    angular_velocity_cost = 0.03 * pitch_rate**2

    # Control regularization. Keep this modest, otherwise the policy becomes passive.
    control_cost = 0.0025 * jnp.sum(action**2)
    smoothness_cost = 0.007 * jnp.sum((action - prev_action) ** 2)

    # Moderate fall penalty. The terminal bonus below rewards full survival.
    # If this is too high, ARS prefers standing; if too low, it accepts falling.
    fall_penalty = 20.0 * (1.0 - alive)

    return alive * (
        healthy_reward
        + forward_reward
        + velocity_tracking
        - no_progress_penalty
        - height_cost
        - posture_cost
        - angular_velocity_cost
        - control_cost
        - smoothness_cost
    ) - fall_penalty


def terminal_locomotion_bonus(qpos_traj, starting_data):
    """End-of-rollout bonus used to distinguish real gait from standing.

    Standing upright is not enough: the survival bonus is only granted if the
    walker survives the whole rollout AND moves at least 1.5 m forward.
    """
    distance = qpos_traj[-1, 0] - starting_data.qpos[0]
    final_height = qpos_traj[-1, 1]
    final_pitch = qpos_traj[-1, 2]
    final_alive = ((final_height > H_MIN) & (jnp.abs(final_pitch) < ANGLE_MAX)).astype(jnp.float32)

    # Encourage progress, but do not reward infinite sprinting too much.
    capped_distance = jnp.clip(distance, 0.0, 9.0)

    distance_bonus = 45.0 * capped_distance
    survived_with_progress_bonus = 900.0 * final_alive * (distance > 1.5).astype(jnp.float32)
    no_gait_penalty = 600.0 * (distance < 1.0).astype(jnp.float32)

    return distance_bonus + survived_with_progress_bonus - no_gait_penalty


# Roll out one linear policy and record reward plus observations.
@jax.jit
def rollout_policy(params, obs_mean, obs_std, starting_data):
    def control_step(carry, _):
        d, prev_action = carry
        obs = walker_obs(d)
        action = policy_action(params, obs, obs_mean, obs_std)
        d0 = d
        d = d.replace(ctrl=action)

        # Hold each action for N_SUBSTEPS fine physics steps.
        def physics_step(inner_d, _):
            return mjx.step(mx_model, inner_d), None

        d, _ = lax.scan(physics_step, d, None, length=N_SUBSTEPS)
        reward = walker_step_reward(d0, d, action, prev_action)
        return (d, action), (reward, walker_obs(d), d.qpos, d.qvel)

    init_action = jnp.zeros(NU)
    _, (rewards, obs_traj, qpos_traj, qvel_traj) = lax.scan(
        control_step, (starting_data, init_action), None, length=HORIZON
    )
    total_reward = jnp.sum(rewards) + terminal_locomotion_bonus(qpos_traj, starting_data)
    return total_reward, obs_traj, qpos_traj, qvel_traj


# Training-specific rollout: returns only reward and observations.
# This is lighter than rollout_policy because ARS/CEM do not need qpos/qvel for every candidate.
@jax.jit
def rollout_reward_obs(params, obs_mean, obs_std, starting_data):
    def control_step(carry, _):
        d, prev_action = carry
        obs = walker_obs(d)
        action = policy_action(params, obs, obs_mean, obs_std)
        d0 = d
        d = d.replace(ctrl=action)

        def physics_step(inner_d, _):
            return mjx.step(mx_model, inner_d), None

        d, _ = lax.scan(physics_step, d, None, length=N_SUBSTEPS)
        reward = walker_step_reward(d0, d, action, prev_action)
        return (d, action), (reward, walker_obs(d), d.qpos)

    init_action = jnp.zeros(NU)
    _, (rewards, obs_traj, qpos_traj) = lax.scan(
        control_step, (starting_data, init_action), None, length=HORIZON
    )
    total_reward = jnp.sum(rewards) + terminal_locomotion_bonus(qpos_traj, starting_data)
    return total_reward, obs_traj


# Evaluate many candidate policies in parallel for training.
@jax.jit
def batched_reward_obs(params_batch, obs_mean, obs_std, starting_data):
    return jax.vmap(rollout_reward_obs, in_axes=(0, None, None, None))(
        params_batch, obs_mean, obs_std, starting_data
    )


@jax.jit
def batched_rewards(params_batch, obs_mean, obs_std, starting_data):
    rewards, _ = batched_reward_obs(params_batch, obs_mean, obs_std, starting_data)
    return rewards


ZERO_MEAN = jnp.zeros(STATE_DIM)
UNIT_STD = jnp.ones(STATE_DIM)

zero_params = jnp.zeros(PARAM_DIM)
zero_reward, zero_obs, zero_qpos, zero_qvel = rollout_policy(zero_params, ZERO_MEAN, UNIT_STD, mx_data)
print("Training horizon:", HORIZON, "control steps =", TRAIN_SECONDS, "s")
print("Policy parameter count:", PARAM_DIM)
print("Zero-policy reward:", float(zero_reward))
print("Observation trajectory shape:", zero_obs.shape)


# %%
# Train a Walker policy with CEM as a baseline attempt.
def train_cem(
    key,
    iterations=12,
    population=256,
    elite_frac=0.10,
    init_std=0.25,
    min_std=0.03,
):
    # CEM needs enough elites to refit a high-dimensional Gaussian.
    elite_count = max(1, int(population * elite_frac))
    mean = jnp.zeros(PARAM_DIM)
    std = jnp.ones(PARAM_DIM) * init_std
    history = []

    best_reward = -float("inf")
    best_params = mean

    for it in range(iterations):
        # Sample a population of policies.
        key, subkey = random.split(key)
        samples = mean + std * random.normal(subkey, (population, PARAM_DIM))

        # Evaluate all policies in parallel.
        rewards = batched_rewards(samples, ZERO_MEAN, UNIT_STD, mx_data)

        # Keep only the elite policies and refit mean/std.
        _, elite_idx = lax.top_k(rewards, elite_count)
        elites = samples[elite_idx]
        mean = jnp.mean(elites, axis=0)
        std = jnp.maximum(jnp.std(elites, axis=0), min_std)

        # Store the best reward for the comparison plot.
        best_idx = int(jnp.argmax(rewards))
        best = float(rewards[best_idx])
        if best > best_reward:
            best_reward = best
            best_params = samples[best_idx]
        history.append(best)
        print(f"CEM iter {it:03d} | best reward {best:8.2f} | overall best {best_reward:8.2f} | mean std {float(jnp.mean(std)):.3f}")

    total_steps = iterations * population * HORIZON * N_SUBSTEPS
    return best_params, jnp.array(history), total_steps, best_reward


# CEM modes. Start with extended_colab. Use fairer_slow only if you still have GPU quota.
CEM_MODE = ARGS.cem_mode

CEM_CONFIGS_BY_MODE = {
    "smoke": {"iterations": 4, "population": 80, "elite_frac": 0.10},
    "extended_colab": {"iterations": 12, "population": 256, "elite_frac": 0.10},
    "fairer_slow": {"iterations": 30, "population": 384, "elite_frac": 0.10},
}

cem_cfg = CEM_CONFIGS_BY_MODE[CEM_MODE]
print("CEM mode:", CEM_MODE, cem_cfg)

key = random.PRNGKey(10)
cem_params, cem_history, cem_steps, cem_best_reward = train_cem(key, **cem_cfg)
print(f"Best CEM reward observed: {cem_best_reward:.2f}")


# %%
# Update running mean and variance for ARS V2 state normalization.
def update_running_stats(mean, var, count, batch):
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - mean
    total_count = count + batch_count
    new_mean = mean + delta * batch_count / total_count

    m_a = var * count
    m_b = batch_var * batch_count
    correction = delta**2 * count * batch_count / total_count
    new_var = (m_a + m_b + correction) / total_count
    return new_mean, jnp.maximum(new_var, 1e-6), total_count


# Train one ARS configuration.
def train_ars(
    key,
    iterations=60,
    directions=128,
    top_frac=0.25,
    step_size=0.03,
    noise=0.05,
):
    # top_b is the number of elite directions used in the ARS update.
    top_b = max(1, int(directions * top_frac))
    params = jnp.zeros(PARAM_DIM)

    # These are the ARS V2 running state-normalization statistics.
    obs_mean = jnp.zeros(STATE_DIM)
    obs_var = jnp.ones(STATE_DIM)
    obs_count = jnp.array(1e-4)
    history = []

    # Keep the best checkpoint, not only the final iteration.
    best_reward = -float("inf")
    best_iter = -1
    best_params = params
    best_norm = (obs_mean, jnp.sqrt(obs_var) + 1e-6)

    for it in range(iterations):
        # Normalize policy observations with the current running statistics.
        obs_std = jnp.sqrt(obs_var) + 1e-6

        # Sample N random directions in policy-parameter space.
        key, subkey = random.split(key)
        deltas = random.normal(subkey, (directions, PARAM_DIM))

        # Evaluate theta + noise * delta and theta - noise * delta.
        candidates = jnp.concatenate(
            [params + noise * deltas, params - noise * deltas], axis=0
        )
        rewards, obs_batch = batched_reward_obs(candidates, obs_mean, obs_std, mx_data)

        # Update the V2 state normalizer using all states from the perturbed rollouts.
        obs_mean, obs_var, obs_count = update_running_stats(
            obs_mean, obs_var, obs_count, obs_batch.reshape((-1, STATE_DIM))
        )

        # Split paired rewards and select top directions by max(r_plus, r_minus).
        rewards_pos = rewards[:directions]
        rewards_neg = rewards[directions:]
        _, elite_idx = lax.top_k(jnp.maximum(rewards_pos, rewards_neg), top_b)

        # Compute the ARS-V1-t update using only elite directions.
        elite_diffs = rewards_pos[elite_idx] - rewards_neg[elite_idx]
        elite_deltas = deltas[elite_idx]
        elite_rewards = jnp.concatenate([rewards_pos[elite_idx], rewards_neg[elite_idx]])
        sigma_r = jnp.std(elite_rewards) + 1e-6
        step = jnp.tensordot(elite_diffs, elite_deltas, axes=1)
        params = params + (step_size / (top_b * sigma_r)) * step

        # Evaluate the updated policy for the learning curve and checkpointing.
        obs_std = jnp.sqrt(obs_var) + 1e-6
        current_reward, _ = rollout_reward_obs(params, obs_mean, obs_std, mx_data)
        current_reward = float(current_reward)
        history.append(current_reward)

        if current_reward > best_reward:
            best_reward = current_reward
            best_iter = it
            best_params = params
            best_norm = (obs_mean, obs_std)

        if it % 5 == 0 or it == iterations - 1:
            print(
                f"ARS iter {it:03d} | reward {current_reward:8.2f} | "
                f"best {best_reward:8.2f} @ {best_iter:03d} | "
                f"sigma_R {float(sigma_r):.3f}"
            )

    total_steps = iterations * (2 * directions + 1) * HORIZON * N_SUBSTEPS
    return best_params, best_norm, jnp.array(history), total_steps, best_reward, best_iter


# Colab-friendly tuning. Start with "smoke", then use "colab_balanced" for the report run.
# Avoid running many configs: each different candidate shape can trigger a new JAX compilation.
# - smoke: verifies that the notebook works quickly
# - colab_balanced: recommended default; should walk more than the over-stable version
# - colab_sweep: two configs if the default still falls or stands
# - strong: heavier run, only if you have enough compute
RUN_MODE = ARGS.run_mode

ARS_CONFIGS_BY_MODE = {
    "smoke": [
        {"name": "smoke_test", "iterations": 20, "directions": 32, "top_frac": 0.25, "step_size": 0.020, "noise": 0.045},
    ],
    "colab_balanced": [
        # Stronger forward drive than the stable version, but with a terminal survival/progress bonus.
        {"name": "colab_balanced", "iterations": 160, "directions": 128, "top_frac": 0.30, "step_size": 0.020, "noise": 0.050},
    ],
    "colab_sweep": [
        {"name": "balanced_safer", "iterations": 140, "directions": 96, "top_frac": 0.35, "step_size": 0.016, "noise": 0.045},
        {"name": "balanced_faster", "iterations": 140, "directions": 96, "top_frac": 0.28, "step_size": 0.022, "noise": 0.055},
    ],
    "strong": [
        {"name": "strong_balanced", "iterations": 240, "directions": 192, "top_frac": 0.30, "step_size": 0.018, "noise": 0.045},
    ],
    "strong_stable": [
        {"name": "strong_stable", "iterations": 280, "directions": 192, "top_frac": 0.35, "step_size": 0.012, "noise": 0.035},
    ],
    "strong_middle": [
        {"name": "strong_middle", "iterations": 260, "directions": 192, "top_frac": 0.32, "step_size": 0.015, "noise": 0.04},
    ],
}

ARS_CONFIGS = ARS_CONFIGS_BY_MODE[RUN_MODE]
print("ARS tuning mode:", RUN_MODE)

ars_results = []
base_key = random.PRNGKey(11)

for idx, cfg in enumerate(ARS_CONFIGS):
    print("\nRunning ARS config:", cfg["name"])
    key = random.fold_in(base_key, idx)
    params, norm, history, steps, best_reward, best_iter = train_ars(
        key,
        iterations=cfg["iterations"],
        directions=cfg["directions"],
        top_frac=cfg["top_frac"],
        step_size=cfg["step_size"],
        noise=cfg["noise"],
    )
    ars_results.append({
        **cfg,
        "params": params,
        "norm": norm,
        "history": history,
        "steps": steps,
        "best_reward": best_reward,
        "best_iter": best_iter,
    })

# Select the best config by the best checkpoint reward, not final reward.
best_idx = int(np.argmax([float(r["best_reward"]) for r in ars_results]))
best_ars = ars_results[best_idx]
ars_params = best_ars["params"]
ars_norm = best_ars["norm"]
ars_history = best_ars["history"]
ars_steps = best_ars["steps"]

print("\nBest ARS config:", best_ars["name"])
print("Best checkpoint reward:", float(best_ars["best_reward"]))
print("Best checkpoint iteration:", int(best_ars["best_iter"]))


# %%
# Compare the extended CEM attempt against every ARS tuning run.
plt.figure(figsize=(10, 6))
plt.plot(np.asarray(cem_history), label=f"CEM attempt ({CEM_MODE})", linewidth=2)

for result in ars_results:
    plt.plot(np.asarray(result["history"]), label=f"ARS {result['name']}")

plt.xlabel("Optimizer iteration")
plt.ylabel("Return")
plt.title("Part 4: Walker2d CEM attempt vs ARS tuning")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(ARGS.output_dir, "learning_curves.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Learning curve saved to:", os.path.join(ARGS.output_dir, "learning_curves.png"))

# Print sample-efficiency numbers for the report.
print(f"CEM approximate physics steps: {cem_steps:,}")
for result in ars_results:
    print(f"ARS {result['name']} approximate physics steps: {result['steps']:,}")

print("\nBest ARS config:")
for key_name in ["name", "iterations", "directions", "top_frac", "step_size", "noise"]:
    print(f"  {key_name}: {best_ars[key_name]}")
print(f"  best checkpoint iteration: {best_ars['best_iter']}")
print(f"  best checkpoint reward: {best_ars['best_reward']:.2f}")

# Diagnose the best checkpoint over the same horizon used for training.
best_reward, best_obs, best_qpos, best_qvel = rollout_policy(
    ars_params, ars_norm[0], ars_norm[1], mx_data
)
best_qpos_np = np.asarray(best_qpos)
best_qvel_np = np.asarray(best_qvel)
heights = best_qpos_np[:, 1]
pitches = best_qpos_np[:, 2]
fallen = (heights < H_MIN) | (np.abs(pitches) > ANGLE_MAX)
first_fall = int(np.argmax(fallen)) if fallen.any() else HORIZON
distance = float(best_qpos_np[-1, 0] - best_qpos_np[0, 0])
mean_vx = float(np.mean(best_qvel_np[:, 0]))

print("\nBest ARS checkpoint diagnostics over training horizon:")
print(f"  return: {float(best_reward):.2f}")
print(f"  distance traveled: {distance:.3f} m")
print(f"  mean forward velocity: {mean_vx:.3f} m/s")
print(f"  min torso height: {float(np.min(heights)):.3f} m")
print(f"  max abs pitch: {float(np.max(np.abs(pitches))):.3f} rad")
print(f"  first fallen step: {first_fall} / {HORIZON}")

if first_fall == HORIZON and distance > 1.5:
    print("\nResult: good. The policy survives the full training horizon and moves forward clearly.")
elif distance > 2.0:
    print("\nResult: partial. The policy walks, but still falls before the end.")
elif first_fall == HORIZON:
    print("\nResult: stable but too passive. Increase forward drive or use RUN_MODE='colab_sweep'.")
else:
    print("\nResult: still unstable. Try RUN_MODE='colab_sweep' or the stronger settings.")


# %%
# Render a trained policy on CPU MuJoCo.
def render_policy_cpu(params, obs_mean, obs_std, steps=500, fps=30, title="ARS Walker2d policy"):
    params = np.asarray(params)
    obs_mean = np.asarray(obs_mean)
    obs_std = np.asarray(obs_std)

    weights = params[: NU * STATE_DIM].reshape(NU, STATE_DIM)
    bias = params[NU * STATE_DIM :]

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)

    renderer = mujoco.Renderer(model, height=360, width=540)
    frames = []
    render_every = max(1, int((1 / fps) / (model.opt.timestep * N_SUBSTEPS)))

    x0 = float(d.qpos[0])
    first_fall = None

    for t in range(steps):
        # Compute the same normalized observation used during ARS training.
        obs = np.concatenate([d.qpos[1:], d.qvel])
        action = np.tanh(weights @ ((obs - obs_mean) / obs_std) + bias)

        # Apply the action for N_SUBSTEPS physics steps.
        d.ctrl[:] = action
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, d)

        # Track fall status during the longer render.
        fallen_now = (d.qpos[1] < H_MIN) or (abs(d.qpos[2]) > ANGLE_MAX)
        if fallen_now and first_fall is None:
            first_fall = t

        # Capture frames at the requested video frame rate.
        if t % render_every == 0:
            renderer.update_scene(d, camera="track")
            frames.append(renderer.render().copy())

    distance = float(d.qpos[0] - x0)
    mean_vx = distance / (steps * DT_CTRL)
    fall_text = f"{first_fall} / {steps}" if first_fall is not None else f"{steps} / {steps}"

    renderer.close()
    print(title)
    print(f"Render duration: {steps * DT_CTRL:.1f} s")
    print(f"Distance during render: {distance:.3f} m")
    print(f"Mean forward velocity during render: {mean_vx:.3f} m/s")
    print(f"First fallen render step: {fall_text}")
    safe_title = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in title.lower())[:80]
    out_video = os.path.join(ARGS.output_dir, safe_title + ".mp4")
    media.write_video(out_video, frames, fps=fps)
    print("Video saved to:", out_video)


# Longer video: this does not increase training cost.
VIDEO_SECONDS = ARGS.video_seconds
render_steps = int(VIDEO_SECONDS / DT_CTRL)

render_policy_cpu(
    ars_params,
    ars_norm[0],
    ars_norm[1],
    steps=render_steps,
    title=f"Best ARS Walker2d policy over {VIDEO_SECONDS}s: {best_ars['name']}",
)

# Optional: render the best CEM policy too. It usually fails or stays poor, which is useful evidence.
RENDER_CEM = ARGS.render_cem
if RENDER_CEM:
    render_policy_cpu(
        cem_params,
        ZERO_MEAN,
        UNIT_STD,
        steps=render_steps,
        title=f"Best CEM Walker2d policy over {VIDEO_SECONDS}s: {CEM_MODE}",
    )


# Save final artifacts for the report and for rerendering without retraining.
metrics = {
    "backend": jax.default_backend(),
    "devices": [str(d) for d in jax.devices()],
    "train_seconds": float(TRAIN_SECONDS),
    "video_seconds": float(VIDEO_SECONDS),
    "cem_mode": str(CEM_MODE),
    "run_mode": str(RUN_MODE),
    "cem_steps": int(cem_steps),
    "cem_best_reward": float(cem_best_reward),
    "ars_steps": int(ars_steps),
    "best_ars_name": str(best_ars["name"]),
    "best_ars_reward": float(best_ars["best_reward"]),
    "best_ars_iter": int(best_ars["best_iter"]),
    "diagnostic_return": float(best_reward),
    "diagnostic_distance_m": float(distance),
    "diagnostic_mean_vx_m_s": float(mean_vx),
    "diagnostic_min_height_m": float(np.min(heights)),
    "diagnostic_max_abs_pitch_rad": float(np.max(np.abs(pitches))),
    "diagnostic_first_fall_step": int(first_fall),
    "diagnostic_horizon_steps": int(HORIZON),
}
with open(os.path.join(ARGS.output_dir, "metrics.json"), "w") as f:
    _json.dump(metrics, f, indent=2)
np.savez(
    os.path.join(ARGS.output_dir, "best_ars_policy.npz"),
    params=np.asarray(ars_params),
    obs_mean=np.asarray(ars_norm[0]),
    obs_std=np.asarray(ars_norm[1]),
    cem_params=np.asarray(cem_params),
    cem_history=np.asarray(cem_history),
    ars_history=np.asarray(ars_history),
)
print("Metrics saved to:", os.path.join(ARGS.output_dir, "metrics.json"))
print("Best policy saved to:", os.path.join(ARGS.output_dir, "best_ars_policy.npz"))
