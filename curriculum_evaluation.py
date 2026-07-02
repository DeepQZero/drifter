"""
curriculum_evaluation.py

Evaluates a chain of curriculum-stage checkpoints in a single continuous
episode. Starting from the hardest stage's model, each episode keeps
stepping until that stage's docking condition is met, then switches to
the next easier model and keeps going, until the easiest stage (model 1)
finishes. Only that final stage's result counts as the episode's win or
loss.

This answers a different question than checkpoint_test.py and the
single-run evaluation flow in nodrift_train.py. Where checkpoint_test.py
checks one model on one difficulty level, curriculum_evaluation.py checks
whether the agent can complete the full mission end to end, chaining
through every curriculum stage.

Supports both drift-assisted (Drifter-Learn) and direct-docking (nodrift)
agents. Set AGENT_MODE at the top to switch between them.

Checkpoint finding and stage extraction are shared with checkpoint_test.py
through evaluation_utilities.py, which is also used by the related
curriculum training and testing scripts.
"""

import os
import contextlib
import io
import time
import numpy as np
from stable_baselines3 import PPO
from evaluation_utilities import extract_stage, resolve_checkpoint_paths

# AGENT MODE
# "drift"    -> uses DriftTestEnv and drift_initial_trainer.get_curriculum
# "nodrift"  -> uses SpaceCraftDockingEnv3D and nodrift_initial_trainer.get_curriculum
AGENT_MODE = "drift"

# Use a fixed seed so test runs are repeatable.
SEED = 0
np.random.seed(SEED)

# How many curriculum-stage checkpoints to chain together in one episode.
NUM_MODELS = 5

# True: print position and velocity norms at each step and every stage switch.
# False: only print per-stage results and the final summary.
VERBOSE = False

# Number of evaluation episodes to run.
NUM_EVAL_EPISODES = 10

# CHECKPOINT CONFIG
# "latest"                            -> newest .zip in the most recent run folder
# "run:safe_PPO_14"                   -> newest .zip inside run folder safe_PPO_14
# "data/checkpoints/safe_PPO_14"      -> newest .zip in this folder
# "pattern:safe_ppo_model_*_*_*.zip"  -> custom glob inside the checkpoint root
# r"C:\...\checkpoint.zip"            -> exact path to one checkpoint file
CHECKPOINT = "run:safe_PPO_14"

# Root folder that contains saved model checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def make_env(mode: str):
    """Create the evaluation environment and curriculum function for the given mode.

    Args:
        mode: "drift" for DriftTestEnv, "nodrift" for SpaceCraftDockingEnv3D.

    Returns:
        A tuple of (env_constructor, get_curriculum_fn, is_docked_fn).

    Raises:
        ValueError: If mode is not "drift" or "nodrift".
    """
    if mode == "drift":
        from drift_env import DriftTestEnv
        from drift_initial_trainer import get_curriculum

        def build_env():
            curriculum, _ = get_curriculum(10)  # stage 10 is the test config
            return DriftTestEnv(**curriculum)

        def is_docked(env):
            return env.env.is_docked()

        def is_drifting(env):
            return env.is_drifting

        return build_env, is_docked, is_drifting

    elif mode == "nodrift":
        from docking_env import SpaceCraftDockingEnv3D
        from nodrift_initial_trainer import get_curriculum, NUM_STAGES

        def build_env():
            # Use the hardest training stage config as the test environment.
            curriculum, _ = get_curriculum(NUM_STAGES - 1)
            # Override to full mission range for the evaluation episode.
            curriculum['max_episode_len'] = 5_000
            curriculum['min_init_pos_bound'] = 75
            curriculum['max_init_pos_bound'] = 150
            return SpaceCraftDockingEnv3D(**curriculum)

        def is_docked(env):
            return env.is_docked()

        def is_drifting(env):
            return False  # nodrift agent never drifts

        return build_env, is_docked, is_drifting

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

print(f"Agent mode: {AGENT_MODE}")
print("Evaluating models:")
for i, p in enumerate(checkpoint_files):
    print(f"  Model {i+1}: {p}")

models = load_and_validate_models(checkpoint_files, CHECKPOINT)
build_env, is_docked_fn, is_drifting_fn = make_env(AGENT_MODE)

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

    env = build_env()
    obs, info = env.reset(seed=SEED + i)
    done = False

    if VERBOSE:
        print(f"  Starting with model {model_num}")

    while not done:
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
            r = np.linalg.norm(obs[0:3])
            v = np.linalg.norm(obs[3:6])
            print(f"  r={r:.3f}, v={v:.3f}")

        done = term or trunc

        if done:
            stage_num = extract_stage(checkpoint_files[model_num - 1])
            stage_result = is_docked_fn(env)
            print(f"  Stage {stage_num} / model {model_num}: {'SUCCESS' if stage_result else 'FAIL'}")

            if model_num == 1:
                # Final stage done — record the episode result.
                wins.append(int(stage_result))
                fuels.append(epi_fuel)
                results["wins"].append(int(stage_result))
                results["fuels"].append(epi_fuel)
                episode_time = time.time() - episode_start
                print(
                    f"  Episode result: {'WIN' if stage_result else 'LOSS'}  "
                    f"fuel: {epi_fuel:.2f}  time: {episode_time:.2f}s"
                )
            else:
                # Chain is not done — switch to the next easier model.
                done = False
                if AGENT_MODE == "drift":
                    env.is_drifting = False
                model_num = max(model_num - 1, 1)

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
