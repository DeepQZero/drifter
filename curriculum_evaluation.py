"""
curriculum_evaluation.py

Evaluates a chain of curriculum-stage checkpoints in a single continuous
episode. Starting from the hardest stage's model, each episode keeps
stepping until that stage's docking condition is met, then switches to
the next easier model and keeps going, until the easiest stage (model 1)
finishes. Only that final stage's result counts as the episode's win or
loss.

This answers a different question than checkpoint_test.py. Where
checkpoint_test.py checks one model on one difficulty level,
curriculum_evaluation.py checks whether the agent can complete the full
mission end to end, chaining through every curriculum stage.

Checkpoint finding and stage extraction are shared with checkpoint_test.py
through evaluation_utilities.py.
"""

import glob
import os
import contextlib
import io
import time

import numpy as np
from stable_baselines3 import PPO

from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from evaluation_utilities import extract_stage, resolve_checkpoint_paths

# Use a fixed seed so test runs are repeatable.
SEED = 0
np.random.seed(SEED)

# How many curriculum-stage checkpoints to chain together in one episode.
NUM_MODELS = 5

# Set to True to print position and velocity norms at each step, plus
# which model the episode started on and every stage switch.
VERBOSE = False

# Run a fixed number of evaluation episodes.
num_eval_episodes = 10

# CHECKPOINT CONFIG
# "latest"                          -> newest .zip in the most recent run folder
# "run:safe_PPO_6"                  -> newest .zip inside run folder safe_PPO_6
# "data/checkpoints/safe_PPO_6"     -> newest .zip in this folder
# "pattern:safe_ppo_model_*_6.zip"  -> custom glob inside the checkpoint root
# r"C:\...\checkpoint.zip"          -> exact path to one checkpoint file
CHECKPOINT = "run:safe_PPO_12"

# Root folder that contains saved model checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def load_and_validate_models(checkpoint_files, checkpoint_spec):
    """Load trained models from checkpoint files and prepare them for evaluation.

    This function:
    1. Checks that we have at least NUM_MODELS files
    2. Sorts them by curriculum stage, lowest to highest
    3. Loads each model into memory on the CPU

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


# Resolve the checkpoint files using the selected strategy.
checkpoint_files = resolve_checkpoint_paths(CHECKPOINT, CHECKPOINT_ROOT, num_models=NUM_MODELS)

# Verify every requested model actually corresponds to a distinct, valid
# stage. This catches three potential problems before any time is spent
# loading models or running episodes: a filename that does not match the
# expected pattern, the same stage number appearing twice (which can
# happen if checkpoints from two different runs got mixed together), and
# asking for more models than there are stages.
stages_found = [extract_stage(p) for p in checkpoint_files]
if -1 in stages_found:
    raise ValueError(f"Could not parse stage number from one or more checkpoint filenames: {checkpoint_files}")
if len(set(stages_found)) != len(stages_found):
    raise ValueError(f"Duplicate stage numbers found in selected checkpoints: {stages_found}")

max_stage = max(stages_found)
if len(checkpoint_files) > max_stage + 1:
    raise ValueError(
        f"Requested {len(checkpoint_files)} models but only stages 0 to {max_stage} exist."
    )

# Make sure at least one checkpoint exists before continuing.
if len(checkpoint_files) < 1:
    raise FileNotFoundError(f"No checkpoints found for spec: {CHECKPOINT}")

# Display the checkpoint files that will be evaluated.
print("Evaluating models:")
for i, p in enumerate(checkpoint_files):
    print(f"  Model {i+1}: {p}")

# Load and validate models.
models = load_and_validate_models(checkpoint_files, CHECKPOINT)

# Store fuel usage and success values across all episodes.
fuels = []
wins = []

# Keep a copy of the results for later reuse if needed.
results = {
    "checkpoints": checkpoint_files,
    "wins": [],
    "fuels": [],
}

# Every episode starts on the hardest stage (the last model in the
# sorted list) and works its way down to model 1.
model_start = NUM_MODELS

for i in range(num_eval_episodes):
    episode_start = time.time()
    np.random.seed(SEED + i)

    print(f"Episode {i+1}/{num_eval_episodes}")

    model_num = model_start
    model = models[model_num - 1]
    if VERBOSE:
        print(f"Starting with model_{model_num}")

    # Track fuel usage for this episode, summed across every stage.
    epi_fuel = 0

    # Create the test curriculum and instantiate the environment. Stage
    # 10 here sets up the test environment itself, not the model's
    # difficulty level, the model used at each step is chosen separately
    # below based on model_num.
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)
    obs, info = env.reset(seed=SEED + i)
    done = False
    t_step = 0

    # Keep stepping until the episode fully completes, possibly chaining
    # through several models along the way.
    while not done:
        # During drift, or during selected time windows, apply zero
        # thrust. This makes the evaluation behavior match the intended
        # test policy.
        if env.is_drifting:
            action = np.array([0.0, 0.0, 0.0])
        else:
            # Ask the current model for the next action.
            expected_obs_size = models[model_num - 1].observation_space.shape[0]
            action = models[model_num - 1].predict(obs[:expected_obs_size], deterministic=True)[0]

        # Track total fuel by accumulating the action magnitude.
        epi_fuel += np.linalg.norm(action)

        # Advance the environment one step. Stdout is suppressed here
        # because the environment itself prints "WIN!" and "UNSAFE!"
        # directly, and this script prints its own clearer per-stage
        # result line below instead.
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, term, trunc, info = env.step(action)
        t_step += 1

        # Print position and velocity norms for debugging and monitoring.
        if VERBOSE:
            r = np.linalg.norm(obs[0:3])
            v = np.linalg.norm(obs[3:6])
            print(f"r={r:.3f}, v={v:.3f}")

        done = term or trunc

        # If the episode ended, either record the final result or switch
        # to the next easier stage.
        if done:
            stage_num = extract_stage(checkpoint_files[model_num - 1])
            stage_result = env.env.is_docked()
            print(f"Stage {stage_num}/model_{model_num}: {'SUCCESS' if stage_result else 'FAIL'}")

            if model_num == 1:
                # The easiest stage just finished, so this is the final
                # result for the whole episode.
                success = stage_result
                wins.append(int(success))
                fuels.append(epi_fuel)

                results["wins"].append(int(success))
                results["fuels"].append(epi_fuel)

                episode_time = time.time() - episode_start
                print(f"  Episode result: {'WIN' if success else 'LOSS'}, fuel used: {epi_fuel:.2f}, evaluation time: {episode_time:.2f}s")
            else:
                # This stage finished but the chain is not done yet.
                # Reset the episode-ending flags and move to the next
                # easier model, then keep stepping in the same
                # environment instance.
                done = False
                env.is_drifting = False
                t_step = 0
                model_num -= 1
                model_num = max(model_num, 1)


# Print a summary of the test run.
print("\n=== TEST RESULTS ===")
print(f"Wins: {sum(wins)}/{num_eval_episodes} ({100 * np.mean(wins):.1f}%)")

print(f"\nFuel Consumption Statistics:")
if len(fuels) > 0:
    print(f"  Mean:   {np.mean(fuels):.2f}")
    print(f"  Median: {np.median(fuels):.2f}")
    print(f"  Min:    {min(fuels):.2f}")
    print(f"  Max:    {max(fuels):.2f}")
else:
    print("  No successful episodes recorded.")

# Sort fuel usage values for percentile reporting.
if len(fuels) > 0:
    fuels_sorted = np.sort(fuels)
    print("\nPercentiles:")
    print(f"  25th: {np.percentile(fuels, 25):.2f}")
    print(f"  50th: {np.percentile(fuels, 50):.2f}")
    print(f"  75th: {np.percentile(fuels, 75):.2f}")
else:
    print("\nPercentiles:")
    print("  No data")

print("\n=== MODELS EVALUATED ===")
for i, p in enumerate(checkpoint_files):
    print(f"  [{i+1}] stage={extract_stage(p)}")
    print(f"      {p}")
    