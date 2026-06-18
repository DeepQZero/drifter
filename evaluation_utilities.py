"""
evaluation_utilities.py

Shared helper functions for the evaluation scripts in this project
(curriculum_evaluation.py, checkpoint_test.py, and future ones).

Putting this logic in one place means a bug fix here applies everywhere,
instead of needing to be copied by hand into each script.

Contents:
1. extract_stage: reads the curriculum stage number from a checkpoint
   filename.
2. resolve_checkpoint_paths: finds checkpoint .zip files based on a simple
   instruction like "latest" or "run:safe_PPO_6".
3. patch_unsafe_termination: fixes episodes ending too early during
   evaluation because of a training-only safety rule.
4. classify_failure: returns a short word describing why an episode
   failed (for example, "crash" or "timeout").
"""

import glob
import os

from numpy.linalg import norm as _norm


def extract_stage(path):
    """
    Read the curriculum stage number out of a checkpoint filename.

    Filenames look like "safe_ppo_model_3_1.zip", meaning stage 3, run 1.
    The pattern is safe_ppo_model_{stage}_{run}.zip, so this pulls out the
    second-to-last number.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        The stage number as an integer. Returns -1 if the filename does
        not match the expected pattern, instead of crashing, so the
        caller can check for that and raise a clear error if needed.
    """
    # Strip the folder path and the .zip ending, then split on "_".
    # "safe_ppo_model_3_1.zip" becomes ["safe", "ppo", "model", "3", "1"]
    filename = os.path.basename(path).replace(".zip", "")
    parts = filename.split("_")

    # The stage number is the second-to-last piece. Returning -1 instead
    # of crashing lets the caller decide how to handle a bad filename.
    try:
        return int(parts[-2])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoint_paths(checkpoint_spec, checkpoint_root, num_models=1):
    """
    Find checkpoint .zip files on disk based on a simple text instruction.

    Supported formats for checkpoint_spec:
        "latest"
            Finds the most recently modified run folder inside
            checkpoint_root, and returns checkpoints from that folder only.

        "run:safe_PPO_6"
            Returns checkpoints from the folder "safe_PPO_6" inside
            checkpoint_root.

        "pattern:safe_ppo_model_*_6.zip"
            Searches every folder inside checkpoint_root for files
            matching this pattern. Use for fine-grained manual control.

        A direct file path
            Returns just that one file.

        A direct folder path
            Returns checkpoints found inside that folder.

    Every pattern below only matches files named like
    "safe_ppo_model_{stage}_{run}.zip". This project also has a second,
    unrelated checkpoint style named like "drifter_ppo_100000_steps.zip",
    which has no stage number or curriculum.

    Args:
        checkpoint_spec: A string describing which checkpoints to find.
        checkpoint_root: The folder that contains all the run folders.
        num_models: How many of the newest matching checkpoints to
            return. Defaults to 1. Use a higher number for a full
            curriculum chain, for example 5 or 11.

    Returns:
        A list of file paths, sorted from lowest to highest stage.

    Raises:
        FileNotFoundError: if no matching files could be found.
        ValueError: if checkpoint_spec does not match a supported format.
    """

    # Case 1: a direct path to one file.
    if os.path.isfile(checkpoint_spec):
        return [checkpoint_spec]

    # Case 2: a direct path to a folder. Search inside it.
    if os.path.isdir(checkpoint_spec):
        search_pattern = os.path.join(checkpoint_spec, "safe_ppo_model_*.zip")
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            raise FileNotFoundError(
                f"No checkpoint .zip files found in directory: {checkpoint_spec}"
            )

        # Sort by stage number, not filename text, so "10" doesn't sort
        # before "2".
        matching_files = sorted(matching_files, key=extract_stage)
        return matching_files[-num_models:]

    # Case 3: "latest". Use only the most recently modified run folder,
    # so the same stage number from two different runs never gets mixed
    # into one result.
    if checkpoint_spec == "latest":
        all_entries = glob.glob(os.path.join(checkpoint_root, "*"))
        run_folders = [entry for entry in all_entries if os.path.isdir(entry)]

        if not run_folders:
            raise FileNotFoundError(
                f"No run folders found under checkpoint root: {checkpoint_root}"
            )

        most_recent_folder = max(run_folders, key=os.path.getmtime)
        search_pattern = os.path.join(most_recent_folder, "safe_ppo_model_*.zip")
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            raise FileNotFoundError(
                f"No curriculum checkpoints found in {most_recent_folder}"
            )

        matching_files = sorted(matching_files, key=extract_stage)
        return matching_files[-num_models:]

    # Case 4: "run:safe_PPO_6", one exact run folder picked by name.
    if checkpoint_spec.startswith("run:"):
        run_folder_name = checkpoint_spec.split("run:")[1]
        search_pattern = os.path.join(
            checkpoint_root, run_folder_name, "safe_ppo_model_*.zip"
        )
        matching_files = glob.glob(search_pattern)
        matching_files = sorted(matching_files, key=extract_stage)
        return matching_files[-num_models:]

    # Case 5: "pattern:...", a custom glob pattern for advanced use.
    if checkpoint_spec.startswith("pattern:"):
        custom_pattern = checkpoint_spec.split("pattern:")[1]
        full_search_pattern = os.path.join(checkpoint_root, "**", custom_pattern)
        matching_files = glob.glob(full_search_pattern, recursive=True)
        matching_files = sorted(matching_files, key=extract_stage)
        return matching_files[-num_models:]

    # Nothing matched, so fail instead of guessing.
    raise ValueError(f"Invalid checkpoint spec: {checkpoint_spec}")


def patch_unsafe_termination(env):
    """
    Turn off an early-termination rule meant for training, not evaluation.

    drift_env.py's rewards() function normally ends an episode the moment
    the spacecraft's speed goes above a limit:

        term = ... or (max(current_speed - speed_limit, 0) > 0)

    and prints "UNSAFE!" when that happens. This is useful during
    training, but during evaluation a model that has not fully learned
    yet (an early curriculum stage, for example) often goes above this
    limit by accident in the first few steps. That ends the episode
    after only 3 or 4 steps, before the agent has any real chance to
    dock, making the model look like it has a 0% success rate even if
    it is learning correctly.

    This function replaces the environment's rewards() function with a
    copy that leaves out that one termination condition and its print
    statement. Every other part of the reward and termination logic,
    docking, crashing, out of bounds, out of fuel, out of time, stays
    identical to drift_env.py's real rewards() function.

    Call this once, right after creating the environment:
        env = DriftTestEnv(**curriculum)
        patch_unsafe_termination(env)

    Args:
        env: A DriftTestEnv (or similar) object. env.env must be the
            inner environment that defines is_docked(), is_crashed(),
            and the other state-checking functions.
    """
    inner_env = env.env  # DriftTestEnv wraps this inner environment

    def _patched_rewards(last_state):
        """Same as drift_env.py's real rewards(), minus the speed-based
        early termination and its "UNSAFE!" print."""
        tot_step_rew = 0

        current_distance = float(_norm(inner_env.state[0:3]))
        prev_distance = float(_norm(last_state[0:3]))
        current_speed = float(_norm(inner_env.state[3:6]))

        docked = inner_env.is_docked()
        crashed = inner_env.is_crashed()
        out_of_time = inner_env.is_out_of_time()
        out_of_fuel = inner_env.is_out_of_fuel()
        out_of_bounds = inner_env.is_out_of_bounds()

        # Speed is intentionally left out of this condition. This is the
        # point of the patch.
        term = docked or crashed or out_of_bounds or out_of_fuel
        trunc = False if term else out_of_time

        if inner_env.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0

        if inner_env.reward_structure == "dense":
            if docked:
                tot_step_rew += 1
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time:
                tot_step_rew += -1

            prox_penalty = inner_env.proximity_penalty_coeff * (
                current_distance - prev_distance
            )
            tot_step_rew += prox_penalty + inner_env.time_penalty

            # The original rewards() prints "UNSAFE!" here when the speed
            # limit is exceeded. That print and the matching termination
            # above are both left out on purpose.

        return tot_step_rew, term, trunc

    inner_env.rewards = _patched_rewards
    print("  [patch] Disabled early speed-based termination for evaluation.")


def classify_failure(env):
    """
    Figure out why an episode ended without docking successfully.

    Call this right after an episode ends, while the environment still
    holds the final state of that episode.

    Args:
        env: A DriftTestEnv (or similar) object, in the state it was in
            right when the episode ended.

    Returns:
        A short string: "crash", "out_of_bounds", "fuel", "timeout", or
        "unknown" if none of those apply.
    """
    inner_env = env.env

    if inner_env.is_crashed():
        return "crash"
    if inner_env.is_out_of_bounds():
        return "out_of_bounds"
    if inner_env.is_out_of_fuel():
        return "fuel"
    if inner_env.is_out_of_time():
        return "timeout"

    # No recognized reason matched; return "unknown" instead of crashing.
    return "unknown"
