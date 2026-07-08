"""
evaluation_utilities.py

Shared helper functions for the evaluation scripts in this project
(curriculum_evaluation.py, checkpoint_test.py, and future ones).

Putting this logic in one place means a bug fix here applies everywhere,
instead of needing to be copied by hand into each script.

Contents:
1. detect_agent_mode / resolve_agent_mode: infer the agent family from a
   checkpoint path, or validate an explicit mode.
2. _folder_run_number: reads the run number from a checkpoint's parent folder.
3. _parse_checkpoint: splits a filename into (run, stage, epoch), for
   all supported formats.
4. extract_stage / extract_run: read the stage or run number from a
   checkpoint filename (drift or nodrift naming).
5. resolve_checkpoint_paths: finds checkpoint .zip files from a simple
   instruction like "latest" or "run:safe_PPO_6".
6. _find_checkpoints_in_folder: internal helper, tries drift, nodrift, and
   fallback patterns in order.
7. patch_unsafe_termination: fixes episodes ending too early during eval
   because of a training-only safety rule.
8. classify_failure: short word for why an episode failed ("crash",
   "timeout"). For DriftTestEnv; see classify_failure_nodrift in
   checkpoint_test.py for nodrift agents.
"""

import re
import glob
import os

from numpy.linalg import norm as _norm

# Matches the run number in a run folder's own name, e.g. "safe_PPO_15"
# -> 15, "nodrift_curriculum_PPO_14" -> 14. This is the base for
# the run number: unlike the numbers embedded in a checkpoint filename,
# the folder name is generally not ambiguous.
_FOLDER_RUN_RE = re.compile(r"_(\d+)$")


def detect_agent_mode(model_path):
    """Detect whether a checkpoint belongs to the drift or nodrift family.

    Args:
        model_path: Full path to a checkpoint file.

    Returns:
        "nodrift" if the path contains "nodrift", otherwise "drift".
    """
    path_lower = model_path.lower()
    if "nodrift" in path_lower:
        return "nodrift"
    return "drift"


def resolve_agent_mode(agent_mode, model_path=None):
    """Resolve the active agent mode, supporting an automatic default.

    Args:
        agent_mode: "drift", "nodrift", or "auto".
        model_path: Checkpoint path used when agent_mode is "auto".

    Returns:
        "drift" or "nodrift".

    Raises:
        ValueError: If agent_mode is invalid or "auto" has no model path.
    """
    if agent_mode in ("drift", "nodrift"):
        return agent_mode

    if agent_mode != "auto":
        raise ValueError(
            f"Unknown AGENT_MODE: {agent_mode!r}. Use 'drift', 'nodrift', or 'auto'."
        )

    if model_path is None:
        raise ValueError("AGENT_MODE='auto' requires a checkpoint path to detect from.")

    return detect_agent_mode(model_path)


def _folder_run_number(path):
    """Read the run number out of a checkpoint's parent folder name.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        The run number as an integer, or -1 if the parent folder name
        does not end in "_<number>".
    """
    folder_name = os.path.basename(os.path.dirname(path))
    match = _FOLDER_RUN_RE.search(folder_name)
    return int(match.group(1)) if match else -1


def _parse_checkpoint(path):
    """Split a checkpoint filename into (run, stage, epoch).

    Three filename types have been used over time, and the
    numbers alone don't say which shape a given file is (all three are
    just plain integers, so trying parts[-2] before parts[-3] as a
    "fallback" can return the wrong number instead of failing):
        1. {prefix}_model_{run}_{stage}_{epoch}.zip   (current format)
        2. {prefix}_model_{stage}_{run}_{epoch}.zip   (previous format,
           stage/run swapped, rename_checkpoints.py migrates it)
        3. {prefix}_model_{stage}_{run}.zip           (earliest format,
           from before epoch checkpoints existed at all, e.g. safe_PPO_1..6:
           one file per stage, no epoch number in the filename)
    To disambiguate 1 vs. 2, this compares each candidate number against
    the run number in the parent folder name (e.g. "safe_PPO_15" -> 15),
    which is unambiguous. Shape 3 is identified by only having two
    trailing numbers instead of three; its epoch is reported as 0 since
    the filename has no epoch to read, matching the "missing epoch"
    default used elsewhere in this file (see get_epoch in
    _find_checkpoints_in_folder).

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        A (run, stage, epoch) tuple of integers. Any field that could not
        be determined is -1.
    """
    filename = os.path.basename(path).replace(".zip", "")
    parts = filename.split("_")
    folder_run = _folder_run_number(path)

    # Collect the trailing numeric parts, e.g.
    # ["safe", "ppo", "model", "12", "6", "9"] -> ["12", "6", "9"]
    trailing = []
    for part in reversed(parts):
        if not part.lstrip("-").isdigit():
            break
        trailing.append(part)
    trailing.reverse()

    if len(trailing) == 3:
        first, second, epoch = (int(p) for p in trailing)
        # Whichever of the two leading numbers matches the folder's own
        # run number is the run; the other is the stage. Default to the
        # current format (run_stage_epoch) if the folder name gives no
        # run number to check against, or if neither number matches it.
        if second == folder_run:
            return second, first, epoch
        return first, second, epoch

    if len(trailing) == 2:
        # Earliest format: {stage}_{run}, no epoch number existed yet.
        # Prefer the folder's own run number, since it's unambiguous;
        # this filename's own run number should already agree with it,
        # but fall back to reading it straight from the filename if the
        # folder name didn't end in "_<number>" for some reason.
        stage, embedded_run = (int(p) for p in trailing)
        run = folder_run if folder_run != -1 else embedded_run
        return run, stage, 0

    return -1, -1, -1


def extract_stage(path):
    """Read the curriculum stage number out of a checkpoint filename.

    See _parse_checkpoint for the filename formats this handles.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        The stage number as an integer. Returns -1 if the filename does
        not match any expected pattern.
    """
    return _parse_checkpoint(path)[1]


def extract_run(path):
    """Read the run number out of a checkpoint filename.

    See _parse_checkpoint for the filename formats this handles.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        The run number as an integer. Returns -1 if the filename does
        not match any expected pattern.
    """
    return _parse_checkpoint(path)[0]


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
            Returns just that one file. Resolved relative to the current
            working directory first, then relative to the project root
            and checkpoint_root, so a path like "data/checkpoints/safe_PPO_12/
            safe_ppo_model_12_7_7.zip" works regardless of where the
            script was launched from.

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

    # If checkpoint_spec looks like a relative path (for example,
    # "data/checkpoints/safe_PPO_12/safe_ppo_model_12_7_7.zip"), it may not
    # resolve correctly if the script was launched from a different working
    # directory. Try the most likely project-local bases before falling
    # through to the isfile/isdir checks below.
    project_root = os.path.dirname(os.path.dirname(checkpoint_root))
    candidate_bases = [project_root, checkpoint_root]
    for base in candidate_bases:
        candidate = os.path.join(base, checkpoint_spec)
        if os.path.isfile(candidate) or os.path.isdir(candidate):
            checkpoint_spec = candidate
            break

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
    # the "9" in "safe_ppo_model_12_6_9.zip".
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
    print("  [Patch] Disabled early speed-based termination for evaluation.")


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
        A short string: "crash", "unsafe", "out_of_bounds", "fuel",
        "timeout", or "unknown" if none of those apply.
    """
    inner_env = env.env

    if inner_env.is_crashed():
        return "crash"
    # drift_env terminates on any speed-limit violation; without this
    # check those episodes would all report as "unknown".
    if inner_env.is_unsafe():
        return "unsafe"
    if inner_env.is_out_of_bounds():
        return "out_of_bounds"
    if inner_env.is_out_of_fuel():
        return "fuel"
    if inner_env.is_out_of_time():
        return "timeout"

    # No recognized reason matched; return "unknown" instead of crashing.
    return "unknown"
