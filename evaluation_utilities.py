"""
evaluation_utilities.py

Shared helper functions for the evaluation scripts in this project
(curriculum_evaluation.py, checkpoint_test.py, and future ones).

Putting this logic in one place means a bug fix here applies everywhere,
instead of needing to be copied by hand into each script.

Contents:
1. extract_stage: reads the curriculum stage number from a checkpoint
   filename. Handles both drift (safe_ppo_model_*) and nodrift
   (nodrift_ppo_model_*) naming conventions.
2. resolve_checkpoint_paths: finds checkpoint .zip files based on a simple
   instruction like "latest" or "run:safe_PPO_6". Supports both checkpoint
   families automatically.
3. _find_checkpoints_in_folder: internal helper that searches a folder for
   checkpoint files, trying drift, nodrift, and fallback patterns in order.
4. patch_unsafe_termination: fixes episodes ending too early during
   evaluation because of a training-only safety rule.
5. classify_failure: returns a short word describing why an episode
   failed (for example, "crash" or "timeout"). For use with DriftTestEnv;
   see classify_failure_nodrift in checkpoint_test.py for nodrift agents.
"""

import glob
import os

from numpy.linalg import norm as _norm


def extract_stage(path):
    """Read the curriculum stage number out of a checkpoint filename.

    Handles two naming conventions used in this project:
        safe_ppo_model_{stage}_{run}_{epoch}.zip
        nodrift_ppo_model_{stage}_{run}_{epoch}.zip
    In both current conventions, stage is the third-to-last number: parts[-3].

    An older two-number drift format, safe_ppo_model_{stage}_{run}.zip
    with no epoch, is also supported as a fallback: stage is parts[-2].

    For other filenames like "final_model.zip" that have no stage number,
    -1 is returned instead of crashing so the caller can decide how to
    handle it.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        The stage number as an integer. Returns -1 if the filename does
        not match any expected pattern.
    """
    # Strip the folder path and the .zip ending, then split on "_".
    # "safe_ppo_model_6_12_9.zip" -> ["safe", "ppo", "model", "6", "12", "9"]
    # "nodrift_ppo_model_4_2_40.zip" -> ["nodrift", "ppo", "model", "4", "2", "40"]
    filename = os.path.basename(path).replace(".zip", "")
    parts = filename.split("_")

    # Current format for both drift and nodrift checkpoints includes an
    # epoch number: {prefix}_model_{stage}_{run}_{epoch}. Stage is third
    # from the end. This is checked first since it is the format currently 
    # used by both initial_trainer.py and nodrift_initial_trainer.py.
    try:
        return int(parts[-3])
    except (IndexError, ValueError):
        pass

    # Fallback for the older two-number drift format with no epoch:
    # safe_ppo_model_{stage}_{run}.zip. Stage is second from the end.
    # TODO: this fallback can probably be removed the old two-number
    # checkpoints are no longer used.
    try:
        return int(parts[-2])
    except (IndexError, ValueError):
        # Neither pattern matched. Return -1 instead of crashing so the
        # caller can decide how to handle an incorrect filename.
        return -1


def resolve_checkpoint_paths(checkpoint_spec, checkpoint_root, num_models=1):
    """Find checkpoint .zip files on disk based on a simple text instruction.

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

    When searching a folder, this function tries three glob patterns in
    order and uses the first one that finds anything:
        1. safe_ppo_model_*.zip     (drift curriculum checkpoints)
        2. nodrift_ppo_model_*.zip  (nodrift curriculum checkpoints)
        3. *.zip                    (fallback, catches final_model.zip etc.)

    This keeps drift and nodrift checkpoint families from being mixed
    together within one search, while still supporting both.

    Args:
        checkpoint_spec: A string describing which checkpoints to find.
        checkpoint_root: The folder that contains all the run folders.
        num_models: How many of the newest matching checkpoints to
            return. Defaults to 1. Use a higher number for a full
            curriculum chain, for example 5 or 11.

    Returns:
        A list of file paths, sorted from lowest to highest stage.

    Raises:
        FileNotFoundError: If no matching files could be found.
        ValueError: If checkpoint_spec does not match a supported format.
    """

    # Case 1: a direct path to one file.
    if os.path.isfile(checkpoint_spec):
        return [checkpoint_spec]

    # Case 2: a direct path to a folder. Search inside it.
    if os.path.isdir(checkpoint_spec):
        matching_files = _find_checkpoints_in_folder(checkpoint_spec)

        if not matching_files:
            raise FileNotFoundError(
                f"No checkpoint .zip files found in directory: {checkpoint_spec}"
            )

        # Sort by stage number, not filename text, so "10" does not sort
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
        matching_files = _find_checkpoints_in_folder(most_recent_folder)

        if not matching_files:
            raise FileNotFoundError(
                f"No checkpoints found in {most_recent_folder}"
            )

        matching_files = sorted(matching_files, key=extract_stage)
        return matching_files[-num_models:]

    # Case 4: "run:safe_PPO_6", one exact run folder picked by name.
    if checkpoint_spec.startswith("run:"):
        run_folder_name = checkpoint_spec.split("run:")[1]
        run_folder_path = os.path.join(checkpoint_root, run_folder_name)
        matching_files = _find_checkpoints_in_folder(run_folder_path)
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


def _find_checkpoints_in_folder(folder):
    """Search a single folder for checkpoint .zip files.

    Tries three glob patterns in order and uses the first one that finds
    anything. If multiple checkpoints exist for the same stage (for
    example when SAVE_ALL_EPOCHS is True during training), only the
    checkpoint with the highest epoch number is kept for each stage.

    Pattern priority:
        1. safe_ppo_model_*.zip     (drift curriculum checkpoints)
        2. nodrift_ppo_model_*.zip  (nodrift curriculum checkpoints)
        3. *.zip                    (fallback for final_model.zip etc.)

    Args:
        folder: Path to the folder to search.

    Returns:
        A list of matching file paths, one per stage. Returns an empty
        list if nothing is found rather than raising an error; the
        caller decides whether an empty result is an error.
    """
    # Try each pattern in order. The "or" chaining means we only fall
    # through to the next pattern if the previous one found nothing.
    files = (
        glob.glob(os.path.join(folder, "safe_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "nodrift_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "*.zip"))
    )

    if not files:
        return []

    # Group every checkpoint file by its stage number. A training run
    # with SAVE_ALL_EPOCHS = True can leave many files for the same
    # stage, one per epoch, so we collect them all first before picking
    # the best one for each stage.
    stage_to_files = {}
    for f in files:
        stage = extract_stage(f)
        if stage not in stage_to_files:
            stage_to_files[stage] = []
        stage_to_files[stage].append(f)

    # For each stage, keep only the checkpoint with the highest epoch
    # number. The epoch is the last number in the filename, for example
    # the "9" in "safe_ppo_model_6_12_9.zip".
    deduplicated = []
    for stage, stage_files in stage_to_files.items():

        def get_epoch(path):
            """Read the epoch number from a checkpoint filename.

            Args:
                path: Full file path to a checkpoint file.

            Returns:
                The epoch number as an integer. Returns 0 if the
                filename has no epoch number, so it sorts first
                rather than crashing the comparison.
            """
            name = os.path.basename(path).replace(".zip", "")
            parts = name.split("_")
            try:
                return int(parts[-1])
            except (IndexError, ValueError):
                return 0

        # Sort the files for this stage by epoch number and keep the
        # last one, which has the highest epoch.
        deduplicated.append(sorted(stage_files, key=get_epoch)[-1])

    return deduplicated


def patch_unsafe_termination(env):
    """Turn off an early-termination rule meant for training, not evaluation.

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
        early termination and its "UNSAFE!" print.

        Args:
            last_state: The state from before this step, used to
                compute the change in distance.

        Returns:
            A tuple of (reward, terminated, truncated), same as the
            original rewards() function.
        """
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

            # Penalize moving away from the chief, same as the real
            # rewards() function.
            prox_penalty = inner_env.dist_coeff * (
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
    """Figure out why an episode ended without docking successfully.

    Call this right after an episode ends, while the environment still
    holds the final state of that episode.

    Works with DriftTestEnv, where the inner environment is accessed
    via env.env. For nodrift (SpaceCraftDockingEnv3D) environments,
    use classify_failure_nodrift in checkpoint_test.py instead, since
    there is no env.env wrapper layer.

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
