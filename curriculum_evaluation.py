"""
curriculum_evaluation.py

Chains curriculum-stage checkpoints into one continuous episode. Starts
on the hardest stage's model, switches to the next easier model each time
the current stage docks, down to the easiest (model 1). Only the final
stage's result counts as a win or loss.

Defaults to the whole curriculum found in a run folder (10 drift stages,
varies for nodrift stages). NUM_MODELS is just a cap, reassigned to the 
actual count after checkpoints are resolved.

On every model switch, apply_stage_fn(env, stage) resets:
    - dock_dist / dock_speed to that stage's own thresholds
    - the timestep counter (state[6]), part of the model's observation
    - fuel_used, so a stage isn't charged for an earlier stage's fuel

checkpoint_test.py checks one model at one difficulty. This script checks
the full mission end to end.

Set AGENT_MODE to switch between drift and nodrift. Checkpoint finding
and per-stage config come from evaluation_utilities.py and each agent's
get_curriculum().
"""

import os
import contextlib
import io
import time
import numpy as np
from stable_baselines3 import PPO
from evaluation_utilities import extract_stage, resolve_checkpoint_paths, resolve_agent_mode

# AGENT MODE
# "drift"    -> uses DriftTestEnv and drift_initial_trainer.get_curriculum
# "nodrift"  -> uses SpaceCraftDockingEnv3D and nodrift_initial_trainer.get_curriculum
# "auto"     -> detects from checkpoint filename: "nodrift" in name -> nodrift, else drift
AGENT_MODE = "auto"

# Use a fixed seed so test runs are repeatable.
SEED = 0
np.random.seed(SEED)

# Max curriculum-stage checkpoints per episode. Set high (e.g., 100) so
# resolve_checkpoint_paths() can cap it to the available checkpoints.
NUM_MODELS = 100

# True: print position and velocity norms at each step and every stage switch.
# False: only print per-stage results and the final summary.
VERBOSE = False

# Number of evaluation episodes to run.
NUM_EVAL_EPISODES = 10

# CHECKPOINT CONFIG
# "latest"                            -> newest .zip in the most recent run folder
# "run:safe_PPO_16"                   -> newest .zip inside run folder safe_PPO_14
# "data/checkpoints/safe_PPO_16"      -> newest .zip in this folder
# "pattern:safe_ppo_model_*_*_*.zip"  -> custom glob inside the checkpoint root
# r"C:\...\checkpoint.zip"            -> exact path to one checkpoint file
CHECKPOINT = "run:safe_PPO_16"  # default: most recent nodrift curriculum run

# Root folder that contains saved model checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def make_env(mode: str):
    """Create the evaluation environment and curriculum function for the given mode.

    Args:
        mode: "drift" for DriftTestEnv, "nodrift" for SpaceCraftDockingEnv3D.

    Returns:
        A tuple of (env_constructor, is_docked_fn, is_drifting_fn,
        apply_stage_fn, cap_timestep_fn).

        apply_stage_fn(env, stage) sets the stage's docking thresholds and
        resets state[6] (timestep) and fuel_used, so each stage is judged on
        its own config rather than values carried from prior stages.
        cap_timestep_fn(env) keeps the timestep in range (drift only).

    Raises:
        ValueError: If mode is not "drift" or "nodrift".
    """
    if mode == "drift":
        from drift_env import DriftTestEnv
        from drift_initial_trainer import get_curriculum

        def build_env(saferl_obs=False):
            curriculum, _ = get_curriculum(10)  # stage 10 is the test config
            return DriftTestEnv(saferl_obs=saferl_obs, **curriculum)

        def is_docked(env):
            return env.env.is_docked()

        def is_drifting(env):
            return env.is_drifting

        def apply_stage(env, stage):
            stage_curriculum, _ = get_curriculum(stage)
            env.env.dock_dist = stage_curriculum['pos_thresh']
            env.env.dock_speed = stage_curriculum['speed_thresh']
            env.env.state[6] = 0
            env.env.fuel_used = 0

        def cap_timestep(env):
            if env.env.state[6] >= 10:
                env.env.state[6] = 0

        return build_env, is_docked, is_drifting, apply_stage, cap_timestep

    elif mode == "nodrift":
        from docking_env import SpaceCraftDockingEnv3D
        from nodrift_initial_trainer import get_curriculum, NUM_STAGES

        def build_env(saferl_obs=False):
            # Use the hardest training stage config as the test environment.
            curriculum, _ = get_curriculum(NUM_STAGES - 1)
            # Override to full mission range for the evaluation episode.
            curriculum['max_episode_len'] = 5_000
            curriculum['min_init_pos_bound'] = 75
            curriculum['max_init_pos_bound'] = 150
            return SpaceCraftDockingEnv3D(saferl_obs=saferl_obs, **curriculum)

        def is_docked(env):
            return env.is_docked()

        def is_drifting(env):
            return False  # Nodrift agent never drifts

        def apply_stage(env, stage):
            stage_curriculum, _ = get_curriculum(stage)
            env.dock_dist = stage_curriculum['pos_thresh']
            env.dock_speed = stage_curriculum['speed_thresh']
            env.state[6] = 0
            env.fuel_used = 0
            env.vel_violation_reward_sum = 0  # Fresh constraint budget per stage

        def cap_timestep(env):
            pass  # Nodrift's long eval episodes never need a mid-stage cap

        return build_env, is_docked, is_drifting, apply_stage, cap_timestep

    else:
        raise ValueError(f"Unknown AGENT_MODE: {mode!r}. Use 'drift' or 'nodrift'.")


def load_and_validate_models(checkpoint_files, checkpoint_spec):
    """Load trained models from checkpoint files and prepare them for evaluation.

    Sorts by curriculum stage (lowest to highest) and loads each model on CPU.

    Args:
        checkpoint_files: List of file paths to model checkpoint files.
        checkpoint_spec: The original specification (used only for error messages).

    Returns:
        A tuple of loaded PPO model objects, sorted by stage.

    Raises:
        FileNotFoundError: If no checkpoint files were provided.
        ValueError: If fewer than NUM_MODELS checkpoint files were found.
    """
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoints found for spec: {checkpoint_spec}")

    if len(checkpoint_files) < NUM_MODELS:
        raise ValueError(
            f"Need at least {NUM_MODELS} checkpoints, got {len(checkpoint_files)}: "
            f"{checkpoint_files}"
        )

    checkpoint_files = sorted(checkpoint_files, key=extract_stage)

    # Use CPU: these small models are generally faster here than on GPU.
    models = []
    for p in checkpoint_files:
        models.append(PPO.load(p, device="cpu"))

    return tuple(models)


# Resolve checkpoint files and validate them before loading any models.
checkpoint_files = resolve_checkpoint_paths(CHECKPOINT, CHECKPOINT_ROOT, num_models=NUM_MODELS)

# NUM_MODELS above is only a cap. Use however many checkpoints were
# actually found so a drift run folder (10 stages) and a nodrift run
# folder (varying stages) both grab their whole curriculum without needing
# a hardcoded count that would either overshoot or leave stages out.
NUM_MODELS = len(checkpoint_files)

stages_found = [extract_stage(p) for p in checkpoint_files]

if -1 in stages_found:
    raise ValueError(
        f"Could not parse stage number from one or more checkpoint filenames: {checkpoint_files}"
    )
if len(set(stages_found)) != len(stages_found):
    raise ValueError(
        f"Duplicate stage numbers found in selected checkpoints: {stages_found}"
    )
max_stage = max(stages_found)
if len(checkpoint_files) > max_stage + 1:
    raise ValueError(
        f"Requested {len(checkpoint_files)} models but only stages 0 to {max_stage} exist."
    )
if len(checkpoint_files) < 1:
    raise FileNotFoundError(f"No checkpoints found for spec: {CHECKPOINT}")

AGENT_MODE = resolve_agent_mode(AGENT_MODE, checkpoint_files[-1])

print(f"Agent mode: {AGENT_MODE}")
print("Evaluating models:")
for i, p in enumerate(checkpoint_files):
    print(f"  Model {i+1}: {p}")

models = load_and_validate_models(checkpoint_files, CHECKPOINT)
build_env, is_docked_fn, is_drifting_fn, apply_stage_fn, cap_timestep_fn = make_env(AGENT_MODE)

# Match the env's observation layout to the models. A 9-element observation
# means they were trained with SafeRL obs on (speed + max_vel_limit appended).
saferl_obs = models[0].observation_space.shape[0] == 9

fuels = []
wins = []
results = {
    "checkpoints": checkpoint_files,
    "agent_mode": AGENT_MODE,
    "wins": [],
    "fuels": [],
}

# Every episode starts on the hardest stage (last model) and works down to model 1.
model_start = NUM_MODELS

for i in range(NUM_EVAL_EPISODES):
    episode_start = time.time()
    np.random.seed(SEED + i)
    print(f"\nEpisode {i+1}/{NUM_EVAL_EPISODES}")

    model_num = model_start
    epi_fuel = 0

    env = build_env(saferl_obs)
    obs, info = env.reset(seed=SEED + i)
    done = False

    # build_env() constructs the environment using the test config, not
    # the hardest model's own stage config, so the docking thresholds
    # and timestep counter must be set here to match model_num's stage
    # before stepping.
    apply_stage_fn(env, extract_stage(checkpoint_files[model_num - 1]))

    if VERBOSE:
        print(f"  Starting with model {model_num}")

    while not done:
        cap_timestep_fn(env)

        if is_drifting_fn(env):
            # During a drift period the agent holds position with zero thrust.
            action = np.array([0.0, 0.0, 0.0])
        else:
            expected_obs_size = models[model_num - 1].observation_space.shape[0]
            action = models[model_num - 1].predict(
                obs[:expected_obs_size], deterministic=True
            )[0]

        # Accumulate fuel in delta-V units to match other full-test scripts.
        if hasattr(env, "env") and hasattr(env.env, "m"):
            epi_fuel += float(np.sum(np.abs(action))) / env.env.m * env.env.step_len
        elif hasattr(env, "m"):
            epi_fuel += float(np.sum(np.abs(action))) / env.m * env.step_len
        else:
            # Fallback: raw action magnitude (not delta-V)
            epi_fuel += np.linalg.norm(action)

        # Suppress environment prints (WIN!, UNSAFE!) so only this script's
        # per-stage result lines are shown.
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, term, trunc, info = env.step(action)

        if VERBOSE:
            # Read from env state (meters), not the possibly normalized obs.
            state = env.env.state if AGENT_MODE == "drift" else env.state
            r = np.linalg.norm(state[0:3])
            v = np.linalg.norm(state[3:6])
            print(f"  r={r:.3f}, v={v:.3f}")

        done = term or trunc

        if done:
            stage_num = extract_stage(checkpoint_files[model_num - 1])
            stage_result = is_docked_fn(env)
            print(f"  Stage {stage_num} / model {model_num}: {'SUCCESS' if stage_result else 'FAIL'}")

            if model_num == 1:
                # Final stage done: record the episode result.
                wins.append(int(stage_result))
                fuels.append(epi_fuel)
                results["wins"].append(int(stage_result))
                results["fuels"].append(epi_fuel)
                episode_time = time.time() - episode_start
                print(
                    f"  Episode result: {'WIN' if stage_result else 'LOSS'}  "
                    f"Fuel: {epi_fuel:.2f}  Eval time: {episode_time:.2f}s"
                )
            else:
                # Chain is not done: switch to the next easier model.
                done = False
                if AGENT_MODE == "drift":
                    env.is_drifting = False
                model_num = max(model_num - 1, 1)
                # Reset docking thresholds and timestep count for the
                # active stage so is_docked() uses the correct range.
                apply_stage_fn(env, extract_stage(checkpoint_files[model_num - 1]))

# Final summary.
print(f"\n{'='*50}")
print(f"TEST RESULTS  ({AGENT_MODE} agent)")
print(f"{'='*50}")
print(f"Wins: {sum(wins)}/{NUM_EVAL_EPISODES} ({100 * np.mean(wins):.1f}%)")

if len(fuels) > 0:
    print(f"\nFuel (delta-V):")
    print(f"  Mean:    {np.mean(fuels):.3f}")
    print(f"  Median:  {np.median(fuels):.3f}")
    print(f"  Min:     {min(fuels):.3f}")
    print(f"  Max:     {max(fuels):.3f}")
    print(f"  25th pct: {np.percentile(fuels, 25):.3f}")
    print(f"  50th pct: {np.percentile(fuels, 50):.3f}")
    print(f"  75th pct: {np.percentile(fuels, 75):.3f}")
else:
    print("  No completed episodes recorded.")

print(f"\nModels evaluated:")
for i, p in enumerate(checkpoint_files):
    print(f"  [{i+1}] stage={extract_stage(p)}  {p}")
