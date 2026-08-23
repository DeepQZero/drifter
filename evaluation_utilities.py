"""Shared helper functions for the evaluation scripts in this project
(curriculum_evaluation.py, checkpoint_test.py, and future ones).

Putting this logic in one place means a bug fix here applies everywhere,
instead of needing to be copied by hand into each script.

Contents:
1. detect_agent_mode / resolve_agent_mode: identify or validate the agent
    type, drift or nodrift.
2. _folder_run_number: read the run number from a checkpoint folder.
3. _parse_checkpoint: split a checkpoint name into (run, stage, epoch).
4. extract_stage / extract_run: read the stage or run from a checkpoint name.
5. resolve_checkpoint_paths: find checkpoints from instructions such as
    "latest" or "run:safe_PPO_6".
6. _find_checkpoints_in_folder: find checkpoints using supported name patterns.
7. patch_unsafe_termination: prevent training-only safety rules from ending
    evaluation episodes too early.
8. classify_failure: identify why a DriftTestEnv episode failed, such as
    "crash" or "timeout". Use classify_failure_nodrift for nodrift agents.
9. save_run_metadata: save the seed, flags, and resolved curriculum in
    run_metadata.json so the run can be reproduced later.
10. resolve_env_config_from_metadata: rebuild the exact environment settings
    recorded for a checkpoint. Handles single-run and curriculum checkpoints.
11. resolve_env_config_best_effort: find environment settings from metadata,
    a safe live curriculum lookup, or a caller-provided fallback.
12. describe_run_type: label a checkpoint as "curriculum" or "no curriculum".
13. unique_checkpoint_path: find an unused checkpoint name when promoting a
    best checkpoint to a new epoch number.
14. center_matplotlib_window: move a plot window to the center of the monitor.
15. plural_word: choose the correct singular or plural word for a count.
15b. format_duration: format seconds as "X min Y sec (Z.ZZ hrs)" for
    trainer output and run metadata.
16. build_discrete_thrust_colormap: color thrust values in clean 0.25 N
    bands. Adds enough bands to include the largest thrust value.
17. interquartile_mean: calculate the mean of the middle 50% of the data.
18. load_run_metadata / summarize_nodrift_run / auto_label_checkpoints:
    compare nodrift checkpoints and label the difference, such as a
    curriculum change, flag change, hyperparameter change, or seed change.
19. resolve_env_flags_from_metadata: read recorded environment flags and
    fill missing flags with False, so evaluation uses the training settings.
20. summarize_drift_run / auto_label_drift_checkpoints: compare drift
    checkpoint chains by their flags and seed. Used by drift_vs_drift_test.py.
21. classify_failure_nodrift: identify why a nodrift episode failed.
22. format_top_failure: format the most common failure, such as "unsafe x5".
23. mean_pairwise_angle_deg: measure how widely episode approach directions
    vary. Near 0 means similar directions; near 90 means broad variation.
24. resolve_drift_stage_thresholds: read the distance and speed targets for
    one drift checkpoint, so each stage uses its trained targets.
24b. resolve_drift_stage_obs_time_norm: read each drift stage's trained
    max_episode_len for scaling the observation timestep.
24c. resolve_drift_eval_config / eval_config_export_fields: resolve the
    shared stage-9 evaluation settings and save their key values in results.
25. resolve_nodrift_env_config: build evaluation settings for one nodrift
    checkpoint and pass them to SpaceCraftDockingEnv3D.
26. resolve_drift_model_chain: find every stage checkpoint in a drift run,
    ordered from hardest stage to easiest.
27. build_thrust_colormap / max_thrust_across_results: use one color scale
    for thrust values across all plotted trajectories.
28. describe_comparison_type: turn a comparison code, such as
    "flag_ablation", into a plain-English explanation.
29. describe_drift_chain: create a short label such as
    "PPO run 25, stage 8 to 0" from a checkpoint chain.
29b. should_show_plot: decide whether a plot window can be displayed,
    based on SHOW_PLOTS and the Matplotlib backend.
30. save_and_show_figure: save a timestamped figure, print its path, and
    show it when the display settings allow.
31. fuel_statistics: calculate the standard L1 and L2 fuel statistics in
    one result dictionary.
32. plot_paired_approach_directions: create the shared approach-direction
    scatter plot for two evaluated agents.
33. print_paired_comparison_table / _COMPARISON_TABLE_ROWS: print the
    shared metric table and fuel percentiles. Missing metrics are skipped.
34. normalize_flag_keys: rename old environment-flag keys to current names
    before comparing checkpoints.
35. plot_fuel_comparison: create a fuel box plot for one or more agents and
    show the individual episode values.
36. get_safe_action: test the model, zero-thrust, and braking actions and
    return the first action that passes the safety check.
37. resolve_and_print_seed: choose and print a seed before loading a model.
38. plot_paired_trajectories_3d: create the shared 3D trajectory plot with
    optional thrust colors and markers.
39. plot_paired_diagnostics: create the shared 2x2 diagnostics plot for
    thrust, distance, and speed.
40. build_export_output_name: create a descriptive ONNX filename from a
    checkpoint's run type, run number, and stage.
41. export_ppo_checkpoint_to_onnx / build_onnxable_policy /
    verify_onnx_export: export a PPO policy to ONNX and verify its output.
42. resolve_obs_flags: identify observation flags from the model's size.
43. write_tidy_csv: write flat dictionaries to a CSV with all keys as columns.
44. export_episode_results_csv: write one CSV row per episode, including
    fuel, length, reward, final speed, outcome, and win status.
45. export_trajectory_steps_csv: write one CSV row per recorded timestep for
    time-series plots.
46. group_env_flags / ungroup_env_flags / _read_env_flags: group or flatten
    environment flags for run_metadata.json and environment construction.
47. get_package_versions: record PPO, Python, and other package versions in
    run metadata. Missing packages are recorded as None.
48. interquartile_mean_ci: calculate a bootstrap confidence interval for IQM.
    fuel_statistics and the comparison table use the result.
48b. paired_difference_ci / mcnemar_exact / print_paired_tests: run paired
    A-versus-B statistical tests using the same start states.
48c. wilson_interval / format_win_rate: calculate and format a win rate with
    its confidence interval.
48c2. apply_flag_config_overrides / propagation_matrices: apply replacement
    flags and provide exact closed-form CWH propagation matrices.
48d. apply_dock_radius_override / warn_if_win_condition_differs: calculate
    effective docking thresholds and warn when two agents use different ones.
48e. print_failure_breakdown: print the count for each failure reason.
49. sample_start_states / save_test_set / load_test_set /
    test_set_fingerprint / resolve_start_states: create, save, load, and
    verify shared evaluation start states.
50. new_safe_action_counts / summarize_safe_action_counts /
    SAFE_ACTION_KINDS: record which safety-check action was selected.
51. wrap_with_action_noise: add uniform action noise to a training environment.
52. clarify_ppo_kl_early_stop_message: make PPO's early-stop message report
    the measured KL trigger correctly.
53. rollout_drift_chain_episode: run one full drift-chain episode, including
    coast, stage handoff, fuel, timing, and the step-limit safeguard.
54. set_equal_3d_axes: give 3D trajectory plots equal, symmetric meter-based
    axes.
55. COAST_THRUST_THRESHOLD / TRUE_COAST_THRESHOLD /
    ACTION_SATURATION_THRESHOLD / APPROACH_CHECK_DISTANCE /
    MAX_STEPS_PER_EPISODE: shared metric and safety-limit constants for all
    evaluation scripts. Plot style and run-size settings stay per script.
56. cap_blas_threads_per_worker: limit each vector-environment worker to one
    BLAS thread to prevent CPU oversubscription.
57. interaction_efficiency: find the cumulative training steps needed to
    reach a dock-rate threshold, 0.8 by default, for curriculum or standalone
    runs.
"""

import re
import glob
import json
import math
import os
import types
from datetime import datetime
from pathlib import Path

# Base for printing checkpoint/metadata paths relative to the repo root.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Matches the run number in a run folder name, e.g. "safe_PPO_15" -> 15.
_FOLDER_RUN_RE = re.compile(r"_(\d+)$")


def resolve_obs_flags(obs_size, space_controls_obs=False):
    """Infer a model's observation flags from its observation size.

    The base size is 7, or 6 when space_controls_obs removes the timestep.
    saferl_obs adds 2 values and custom_braking_margin_obs adds 1.
    Pass space_controls_obs from checkpoint metadata because sizes 7-9 can
    match either layout. Missing metadata defaults to False.

    Args:
        obs_size: model.observation_space.shape[0] for a loaded PPO model.
        space_controls_obs: Whether the timestep was dropped from the
            observation. Defaults to False.

    Returns:
        A (saferl_obs, custom_braking_margin_obs) tuple.

    Raises:
        ValueError: If obs_size does not match a supported flag combination.
    """
    base = 6 if space_controls_obs else 7
    layouts = {
        base: (False, False),
        base + 1: (False, True),
        base + 2: (True, False),
        base + 3: (True, True),
    }
    if obs_size not in layouts:
        raise ValueError(
            f"Observation size {obs_size} is not reachable with "
            f"space_controls_obs={space_controls_obs} (valid: "
            f"{base} to {base + 3}). Either the checkpoint's "
            f"space_controls_obs value is wrong, or this is a legacy "
            f"checkpoint whose observation layout predates the current one."
        )
    return layouts[obs_size]


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

    Three filename shapes have been used over time:
        1. {prefix}_model_{run}_{stage}_{epoch}.zip   (current)
        2. {prefix}_model_{stage}_{run}_{epoch}.zip   (previous, swapped)
        3. {prefix}_model_{stage}_{run}.zip           (earliest, no epoch)
    To tell 1 from 2, this checks which candidate number matches the run
    number in the parent folder name (e.g. "safe_PPO_15" -> 15). Shape 3
    is detected by having only two trailing numbers; its epoch reads as 0.

    Args:
        path: Full file path to a checkpoint file.

    Returns:
        A (run, stage, epoch) tuple of integers. Unknown fields are -1.
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
        # Whichever leading number matches the folder's run number is the
        # run; the other is the stage. Defaults to run_stage_epoch order
        # if neither number matches.
        if second == folder_run:
            return second, first, epoch
        return first, second, epoch

    if len(trailing) == 2:
        # Earliest format: {stage}_{run}, no epoch number yet. Prefers the
        # folder's run number, falls back to the filename's.
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


def resolve_checkpoint_paths(checkpoint_spec, checkpoint_root, num_models=1, folder_prefix=None):
    """Find checkpoint .zip files from a simple text instruction.

    Supported forms are "latest", "run:safe_PPO_6", "pattern:*_6.zip",
    a file path, or a folder path. Folder searches use known drift and
    nodrift filename patterns without mixing checkpoint families.

    Args:
        checkpoint_spec: A string describing which checkpoints to find.
        checkpoint_root: The folder that contains all the run folders.
        num_models: Number of newest matching checkpoints to return.
        folder_prefix: Optional prefix, or tuple of prefixes, for "latest".

    Returns:
        A list of file paths, sorted from lowest to highest stage.

    Raises:
        FileNotFoundError: If no matching files could be found.
        ValueError: If checkpoint_spec does not match a supported format.
    """

    # A relative path may not resolve if launched from a different working
    # directory, so try the likely project-local bases first.
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

    # Case 3: "latest". Use the run folder with the highest run number.
    if checkpoint_spec == "latest":
        all_entries = glob.glob(os.path.join(checkpoint_root, "*"))
        run_folders = [entry for entry in all_entries if os.path.isdir(entry)]
        if folder_prefix is not None:
            run_folders = [
                entry for entry in run_folders
                if os.path.basename(entry).startswith(folder_prefix)
            ]

        if not run_folders:
            prefix_msg = f" matching prefix {folder_prefix!r}" if folder_prefix else ""
            raise FileNotFoundError(
                f"No run folders{prefix_msg} found under checkpoint root: {checkpoint_root}"
            )

        def _latest_sort_key(folder_path):
            # Run number first (unparseable names get -1); mtime only
            # breaks ties between folders sharing a run number.
            match = _FOLDER_RUN_RE.search(os.path.basename(folder_path))
            run_number = int(match.group(1)) if match else -1
            return (run_number, os.path.getmtime(folder_path))

        most_recent_folder = max(run_folders, key=_latest_sort_key)
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
    """Find one checkpoint per stage in a folder.

    Uses supported filename patterns in order and keeps the highest epoch
    when a stage has multiple checkpoints.

    Args:
        folder: Path to the folder to search.

    Returns:
        A list of matching file paths, one per stage, or an empty list.
    """
    # Try each pattern in order. The "or" chaining means we only fall
    # through to the next pattern if the previous one found nothing.
    files = (
        glob.glob(os.path.join(folder, "safe_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "nodrift_curriculum_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "nodrift_standalone_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "nodrift_ppo_model_*.zip")) or
        glob.glob(os.path.join(folder, "*.zip"))
    )

    if not files:
        return []

    # Group checkpoint files by stage number. SAVE_ALL_EPOCHS=True can
    # leave multiple files per stage, one per epoch.
    stage_to_files = {}
    for f in files:
        stage = extract_stage(f)
        if stage not in stage_to_files:
            stage_to_files[stage] = []
        stage_to_files[stage].append(f)

    # Keep only the highest-epoch checkpoint per stage (epoch is the
    # last number in the filename).
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

    During evaluation, a new model may exceed the speed limit immediately.
    This wrapper disables that early stop only while rewards() runs. Other
    reward and termination checks, including the drift safety check, remain.

    Call this once after creating the environment:
        env = DriftTestEnv(**curriculum)
        patch_unsafe_termination(env)

    Args:
        env: A DriftTestEnv (or similar) object. env.env must be the
            inner environment that defines rewards() and is_unsafe().

    Returns:
        The same env, patched in place, so calls can be chained.
    """
    inner_env = env.env  # DriftTestEnv wraps this inner environment
    # The unbound class function, not inner_env.rewards, so types.MethodType
    # below resolves "self" at call time. The lookahead paths deepcopy the
    # env, and deepcopy rebinds a MethodType to the copy but not a closure.
    rewards_impl = type(inner_env).rewards

    def _patched_rewards(self, last_state):
        """Delegate to the real rewards() with is_unsafe() disabled."""
        # Shadow is_unsafe only during this call, then remove the shadow.
        self.is_unsafe = lambda: False
        try:
            return rewards_impl(self, last_state)
        finally:
            del self.is_unsafe

    inner_env.rewards = types.MethodType(_patched_rewards, inner_env)
    print("  [Patch] Disabled early speed-based termination for evaluation.")


SAFE_ACTION_KINDS = ("model", "zero_thrust", "braking", "none_safe")


def new_safe_action_counts():
    """Fresh tally of which candidate get_safe_action() picked.

    Keys are SAFE_ACTION_KINDS: "model" when the policy's own action was
    already safe, "zero_thrust" and "braking" when it was overridden, and
    "none_safe" when no candidate passed and the original was returned.

    Returns:
        A dict of counts, all zero.
    """
    return {kind: 0 for kind in SAFE_ACTION_KINDS}


def summarize_safe_action_counts(counts):
    """Turn a safe-action tally into fractions of checked steps.

    Measures how often the learned policy was already safe versus needed
    overriding, which says whether the shield is active or is
    mostly decoration.

    Args:
        counts: A dict from new_safe_action_counts(), after a run.

    Returns:
        A dict with "safe_action_checked_steps" (total) and one
        "safe_action_<kind>_fraction" per kind. All fractions are 0.0
        when no step was ever checked.
    """
    total = sum(counts.get(kind, 0) for kind in SAFE_ACTION_KINDS)
    summary = {"safe_action_checked_steps": total}
    for kind in SAFE_ACTION_KINDS:
        share = counts.get(kind, 0) / total if total else 0.0
        summary[f"safe_action_{kind}_fraction"] = share
    return summary


def get_safe_action(env, action, counts=None, events=None, step=None, episode=None):
    """Pick a safe action, falling back to zero thrust or a braking action.

    Tests the model action, zero thrust, and braking in that order.
    It returns the first candidate that passes a one-step safety check.
    The check uses a copied environment, so observation flags do not affect it.

    Args:
        env: The current DriftTestEnv instance.
        action: The action the model predicted for this step.
        counts: Optional dict from new_safe_action_counts(), incremented
            in place to record which candidate was chosen. Left alone
            when None, so callers that do not care are unaffected.
        events: Optional list, appended to with one dict per checked
            step recording the flight state the decision was made in
            (see below). counts answers "which candidate was used most";
            events additionally answers "when, and in what situation",
            which counts alone cannot. Left alone when None.
        step: The caller's step index within the episode, recorded on the
            event so overrides can be located in time (early approach
            versus terminal braking). Only used when events is given.
        episode: The caller's episode index, recorded so events can be
            grouped per episode. Only used when events is given.

    Returns:
        The first candidate action found to be safe, or the original
        action if none of the candidates are safe.
    """
    import copy  # local import: keeps this module lightweight for path-only callers
    import numpy as np

    # Bounded to the per-axis thruster limit. The braking candidate comes
    # from the velocity vector, which can exceed u_max, and the env does
    # not bound the action it is given.
    u_max = env.env.u_max
    candidates = (
        ("model", np.clip(action, -u_max, u_max)),
        ("zero_thrust", np.array([0.0, 0.0, 0.0])),
        ("braking", np.clip(-1 * env.env.state[3:6], -u_max, u_max)),
    )

    def _record(kind):
        """Tally the chosen candidate, and log its flight context."""
        if counts is not None:
            counts[kind] = counts.get(kind, 0) + 1
        if events is None:
            return
        # Read the pre-decision state, so distance/speed describe the
        # situation the shield reacted to rather than the result.
        distance = float(np.linalg.norm(env.env.state[0:3]))
        speed = float(np.linalg.norm(env.env.state[3:6]))
        speed_limit = float(env.env.velocity_limit(distance))
        events.append({
            "episode": episode,
            "step": step,
            "kind": kind,
            # True when the shield replaced the policy's own action.
            "overridden": kind != "model",
            "distance": distance,
            "speed": speed,
            "speed_limit": speed_limit,
            # Positive means already over the distance-scaled limit.
            "speed_margin": speed - speed_limit,
        })

    for kind, tmp_action in candidates:
        tmp_env = copy.deepcopy(env)
        tmp_env.step(tmp_action)
        if not tmp_env.env.is_unsafe():
            _record(kind)
            return tmp_action

    print("NO SAFE ACTION")
    _record("none_safe")
    return action


def resolve_and_print_seed(seed_setting, leading_newline=False):
    """Resolve a SEED setting to a concrete value and print it.

    None selects a fresh seed. Call this before loading a model because
    SB3 model loading can reset NumPy's random generator.

    Args:
        seed_setting: The caller's SEED constant: an int, or None to draw
            a fresh random seed.
        leading_newline: True prints a blank line before "Seed: ...", to
            match scripts that visually separate this from prior output.

    Returns:
        The resolved integer seed, ready to pass to env.reset(seed=...).
    """
    import numpy as np  # local import: keeps this module lightweight for path-only callers

    prefix = "\n" if leading_newline else ""
    if seed_setting is None:
        resolved_seed = int(np.random.randint(0, 1_000_000))
        print(f"{prefix}Seed: {resolved_seed} (randomly chosen; set SEED to this value to reproduce)")
    else:
        resolved_seed = seed_setting
        print(f"{prefix}Seed: {resolved_seed}")
    return resolved_seed


def classify_failure(env):
    """Figure out why a DRIFT episode ended without docking.

    Call right after the episode ends, while env still holds its final
    state. DriftTestEnv wraps the real environment one layer deep, so
    this reads env.env. For nodrift, use classify_failure_nodrift().

    Args:
        env: A DriftTestEnv (or similar) object at episode end.

    Returns:
        "crash", "unsafe", "out_of_bounds", "fuel", "timeout", or
        "unknown".
    """
    inner_env = env.env

    if inner_env.is_crashed():
        return "crash"
    # _unsafe_is_terminal(), not is_unsafe(): the two agree under the
    # default constraint, but with the budgeted constraint on, is_unsafe()
    # is True on any violating step while only budget exhaustion ends the
    # episode. This is the same predicate rewards() terminates on.
    if inner_env._unsafe_is_terminal():
        return "unsafe"
    if inner_env.is_out_of_bounds():
        return "out_of_bounds"
    if inner_env.is_out_of_fuel():
        return "fuel"
    if inner_env.is_out_of_time():
        return "timeout"
    return "unknown"


def classify_failure_nodrift(env):
    """Figure out why a NODRIFT episode ended without docking.

    The nodrift twin of classify_failure(). Two differences: no wrapper
    layer, so this reads env directly; and "unsafe" comes from
    vel_budget_exhausted() (violations accumulate against a budget)
    rather than is_unsafe() (drift ends on the first violation).

    Args:
        env: A SpaceCraftDockingEnv3D object at episode end.

    Returns:
        "crash", "unsafe", "out_of_bounds", "fuel", "timeout", or
        "unknown".
    """
    if env.is_crashed():
        return "crash"
    if env.vel_budget_exhausted():
        return "unsafe"
    if env.is_out_of_bounds():
        return "out_of_bounds"
    if env.is_out_of_fuel():
        return "fuel"
    if env.is_out_of_time():
        return "timeout"
    return "unknown"


def rollout_drift_chain_episode(env, obs, models, thresholds, model_paths, *,
                                max_steps=5000, use_safe_action=False,
                                safe_action_counts=None, safe_action_events=None,
                                episode_idx=0, use_action_noise=False,
                                print_thrust=False, print_thrust_every=50,
                                verbose=False, print_stage_results=False,
                                stage_obs_time_norms=None,
                                final_pos_override=None, final_speed_override=None):
    """Run one full drift-chain episode and return its recorded data.

    Handles coast periods, stage handoffs, fuel, timing, and the step limit.
    Shared by the drift evaluation scripts so they use the same episode loop.

    Args:
        env: A DriftTestEnv, already reset (fixed_start/fixed_state set
            and reset() called) so it is at t=0 of this episode.
        obs: The observation env.reset() returned.
        models: Chain of loaded PPO models, hardest (index 0) first.
        thresholds: (pos_thresh, speed_thresh) per model, same order.
        model_paths: Checkpoint path per model, same order. Used for
            optional stage output and safety-check events.
        max_steps: Safety limit. Reaching it records a timeout loss.
        use_safe_action: True replaces each model action with
            get_safe_action()'s one-step lookahead safety check.
        safe_action_counts: Optional dict from new_safe_action_counts(),
            passed straight through to get_safe_action() when
            use_safe_action is True.
        safe_action_events: Optional list, passed straight through to
            get_safe_action() when use_safe_action is True.
        episode_idx: This episode's index, recorded on safe-action
            events and printed diagnostics.
        use_action_noise: True multiplies each action by uniform noise
            in [0.95, 1.05] before stepping.
        print_thrust: True prints a per-step thrust/distance/speed line
            every print_thrust_every steps.
        print_thrust_every: See print_thrust.
        verbose: True prints the raw distance/speed every step (matches
            drift_full_test.py's VERBOSE) and, together with
            print_stage_results, the per-stage WIN/FAIL line.
        print_stage_results: True prints a "Model N/M (stage S): WIN/FAIL"
            line at every stage handoff. Kept separate from verbose since
            drift_vs_drift_test.py and drift_vs_nodrift_test.py gate this
            on their own PRINT_STAGE_RESULTS flag, not VERBOSE.
        stage_obs_time_norms: Optional per-model timestep divisors. When
            provided, each stage uses the value from its training run.
            None leaves the evaluation environment's value unchanged.
        final_pos_override: Final-stage docking distance override, in
            meters. None keeps the trained value.
        final_speed_override: Final-stage docking speed override, in m/s.
            None keeps the trained value.

    Returns:
        A dict:
            "win": bool, whether the full chain reached its final dock.
            "outcome": "win" or "loss".
            "failure_type": classify_failure()'s result, or None on a win.
            "episode_length": total step count across the whole chain.
            "elapsed_seconds": real simulated time across the whole
                chain, in seconds: episode_length * step_len. Every
                flown step is step_len seconds long, whether or not the
                coast hold was active, because DriftTestEnv.step() never
                passes drift=True to the inner env (drift_step_len
                applies only inside _det_drift()'s deep-copied
                lookahead). Reported alongside episode_length so fuel
                and time-to-dock can be compared on the same footing as
                nodrift, whose steps are also step_len seconds. step_len
                is read once at the end, not tracked per step, since it
                never changes mid-chain (only dock_dist/dock_speed do,
                at each stage handoff).
            "fuel": accumulated L2 Delta-v (vector norm, secondary metric).
            "fuel_l1": accumulated L1 Delta-v (per-axis sum, primary metric).
            "episode_reward": summed env.step() reward across the episode.
            "final_state": env.env.state, copied, at episode end.
            "positions": list of state[0:3] copies, one per step plus the
                initial state (length episode_length + 1).
            "actions": per-step raw 3-element action vector, same indexing
                as positions (index 0 is [0, 0, 0]). Kept alongside
                thrusts/thrusts_l1 so a caller needing something neither
                norm captures (e.g. per-axis saturation) does not have to
                reconstruct it.
            "thrusts": per-step L2 thrust magnitude, same indexing as
                positions (index 0 is 0.0, the initial no-action state).
            "thrusts_l1": per-step L1 thrust magnitude, same indexing.
            "speeds": per-step speed (L2 velocity norm), same indexing.
            "speed_limits": per-step velocity_limit() at that step's
                distance, same indexing. Recorded here rather than left
                to the caller because velocity_limit() depends on
                dock_dist (under space_controls_speed_limit_offset), and
                dock_dist changes at every stage handoff, so it cannot be
                correctly recomputed after the episode ends.
            "is_drifting": per-step bool, whether the coast hold was
                active for that step (index 0 is False, no coast at episode start).
    """
    import contextlib
    import io
    import numpy as np

    fuel = 0.0
    fuel_l1 = 0.0
    episode_reward = 0.0
    positions = [env.env.state[0:3].copy()]
    actions = [np.array([0.0, 0.0, 0.0])]
    thrusts = [0.0]
    thrusts_l1 = [0.0]
    speeds = [float(np.linalg.norm(env.env.state[3:6]))]
    is_drifting_trace = [False]
    speed_limits = [float(env.env.velocity_limit(
        float(np.linalg.norm(env.env.state[0:3]))))]

    def _apply_stage_obs_time_norm(idx):
        """Point obs[6]'s divisor at this stage's own trained value."""
        if stage_obs_time_norms is None:
            return
        norm = stage_obs_time_norms[idx]
        if norm:
            env.env.obs_time_norm = norm

    def _apply_final_dock_override(idx):
        """Force the final stage's win condition, if idx is that stage."""
        if idx != len(models) - 1:
            return
        if final_pos_override is None and final_speed_override is None:
            return
        env.env.dock_dist, env.env.dock_speed = override_dock_threshold(
            env.env.dock_dist, env.env.dock_speed,
            final_pos_override, final_speed_override, verbose=False,
        )

    model_idx = 0
    model = models[model_idx]
    env.env.dock_dist, env.env.dock_speed = thresholds[model_idx]
    _apply_stage_obs_time_norm(model_idx)
    _apply_final_dock_override(model_idx)
    if stage_obs_time_norms is not None:
        # obs was built by the caller's reset(), before the divisor above
        # was applied, so rebuild it or the first step runs on the old scale.
        obs = env.env._get_obs()

    done = False
    step_count = 0
    result = None

    while not done:
        if env.is_drifting:
            # During a drift period the agent holds position with zero thrust.
            action = np.array([0.0, 0.0, 0.0])
        else:
            # Slice obs to match the observation size the model was trained on.
            # Older models expect 6 elements; newer ones expect 7 (with timestep).
            expected_obs_size = model.observation_space.shape[0]
            action = model.predict(obs[:expected_obs_size], deterministic=True)[0]
            if use_safe_action:
                action = get_safe_action(env, action, safe_action_counts,
                                         events=safe_action_events,
                                         step=step_count, episode=episode_idx)

        if use_action_noise:
            action = action * np.random.uniform(0.95, 1.05)

        # thrust_mag_l1 (per-axis sum) is the primary fuel metric (propellant used).
        # thrust_mag (Euclidean norm) is a secondary metric (gimbaled-thruster equivalent).
        # Both are before mass/step scaling.
        thrust_mag = float(np.linalg.norm(action))
        thrust_mag_l1 = float(np.sum(np.abs(action)))

        # Accumulate fuel as thrust/mass*step_len, matching drift_env.py's
        # fuel_used (L2) and fuel_used_l1 (L1). Summed manually since
        # fuel_used resets on each model switch.
        fuel += thrust_mag / env.env.m * env.env.step_len
        fuel_l1 += thrust_mag_l1 / env.env.m * env.env.step_len

        # Suppress stdout here: the env itself prints "WIN!" on a successful
        # dock, which would duplicate the per-stage result line below.
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, term, trunc, info = env.step(action)
        episode_reward += reward

        if verbose:
            print(np.linalg.norm(env.env.state[0:3]), np.linalg.norm(env.env.state[3:6]))
        if print_thrust and step_count % print_thrust_every == 0:
            pos_norm = np.linalg.norm(env.env.state[0:3])
            speed_norm = np.linalg.norm(env.env.state[3:6])
            print(f'    t={step_count:4d}  thrust={np.round(action, 3)}  '
                  f'|thrust|={thrust_mag:.3f}  dist={pos_norm:.2f}m  speed={speed_norm:.3f}m/s'
                  f'{"  DRIFTING" if env.is_drifting else "  THRUSTING"}')

        done = term or trunc
        step_count += 1

        positions.append(env.env.state[0:3].copy())
        actions.append(np.asarray(action, dtype=float).copy())
        thrusts.append(thrust_mag)
        thrusts_l1.append(thrust_mag_l1)
        speeds.append(float(np.linalg.norm(env.env.state[3:6])))
        is_drifting_trace.append(env.is_drifting)
        # Recorded per step, not recomputed later: velocity_limit() reads
        # dock_dist, which changes at every stage handoff, so a post-loop
        # pass would score all steps against the final stage's value.
        speed_limits.append(float(env.env.velocity_limit(
            float(np.linalg.norm(env.env.state[0:3])))))

        if step_count >= max_steps and not done:
            # Hit the safety limit without a real ending. Record it as a
            # timeout loss rather than dropping the episode entirely.
            result = {"win": False, "outcome": "loss", "failure_type": "timeout"}
            break

        if done:
            docked = env.env.is_docked()
            stage_result = 'WIN' if docked else 'FAIL'
            if verbose or print_stage_results:
                stage_num = extract_stage(str(model_paths[model_idx]))
                print(f'  Model {model_idx + 1}/{len(models)} (stage {stage_num}): {stage_result}')

            if docked and model_idx != len(models) - 1:
                # Stage docked but the chain isn't done: reset the episode-ending
                # flags and switch to the next easier model, then keep stepping.
                done = False
                env.is_drifting = False
                env.env.state[6] = 0  # reset the timestep counter in the state vector
                # Fuel is tracked locally in fuel/fuel_l1 above (accumulated
                # across the whole episode), so env.env.fuel_used doesn't need resetting here.
                model_idx += 1
                model = models[model_idx]
                env.env.dock_dist, env.env.dock_speed = thresholds[model_idx]
                _apply_stage_obs_time_norm(model_idx)
                _apply_final_dock_override(model_idx)
                if stage_obs_time_norms is not None:
                    # Rebuild obs so the incoming stage's first step sees the
                    # zeroed timestep under its own divisor. Only when the
                    # caller opted in, leaving the default path unchanged.
                    obs = env.env._get_obs()
            else:
                # Either the final stage ended, or a mid-chain stage failed some
                # other way (crash, out of bounds, fuel, unsafe, timeout). Either
                # way the episode is over; a mid-chain crash counts as a real loss.
                result = {
                    "win": docked,
                    "outcome": "win" if docked else "loss",
                    "failure_type": None if docked else classify_failure(env),
                }

    # Every flown step advances by step_len, coasting ones included:
    # DriftTestEnv.step() never passes drift=True. drift_step_len applies
    # only inside _det_drift()'s deep-copied lookahead.
    elapsed_seconds = step_count * env.env.step_len

    result.update({
        "episode_length": step_count,
        "elapsed_seconds": elapsed_seconds,
        "fuel": fuel,
        "fuel_l1": fuel_l1,
        "episode_reward": episode_reward,
        "final_state": env.env.state.copy(),
        "positions": positions,
        "actions": actions,
        "thrusts": thrusts,
        "thrusts_l1": thrusts_l1,
        "speeds": speeds,
        "speed_limits": speed_limits,
        "is_drifting": is_drifting_trace,
    })
    return result


def wilson_interval(successes, total, confidence=0.95):
    """Confidence interval for a win rate, accounting for sample size.

    A bare percentage (e.g. "100%" from 30/30) overstates how precisely
    the true win rate is known. The Wilson score interval gives a
    realistic range instead, e.g. 100% at n=30 is really "somewhere in
    [88.6%, 100%]", not a proven perfect policy.

    Args:
        successes: Number of wins.
        total: Number of episodes.
        confidence: 0.90, 0.95, or 0.99.

    Returns:
        A (low, high) tuple, both in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence)
    if z is None:
        raise ValueError(f"Unsupported confidence level: {confidence}")
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_win_rate(wins, total):
    """Format a win rate with its 95% Wilson confidence interval.

    Args:
        wins: Number of wins.
        total: Number of episodes.

    Returns:
        A string like "27/30 (90.0%) [74.4, 96.5]".
    """
    win_rate = wins / total if total else 0.0
    lo, hi = wilson_interval(wins, total)
    return f"{wins}/{total} ({100 * win_rate:.1f}%) [{100 * lo:.1f}, {100 * hi:.1f}]"


def format_top_failure(failure_counts):
    """Turn a failure tally into one short line for a results table.

    Picks whichever failure happened most often and formats it as
    "reason xN", for example "unsafe x5". When nothing failed, returns
    "N/A" instead of naming a reason with a count of zero.

    Args:
        failure_counts: A dict mapping a failure reason to how many times
            it happened, e.g. {"crash": 0, "unsafe": 5, "timeout": 1}.

    Returns:
        A string like "unsafe x5", or "N/A" when every count is zero.
    """
    top_failure = max(failure_counts, key=failure_counts.get)
    top_count = failure_counts[top_failure]
    return "N/A" if top_count == 0 else f"{top_failure} x{top_count}"


def print_failure_breakdown(label, failure_counts):
    """Print every failure reason that happened at least once.

    format_top_failure() only names the single most common reason, so a
    checkpoint with, say, 2 crashes and 1 timeout would only show
    "crash x2" and lose the timeout entirely. This prints the full tally
    instead, one line per nonzero reason.

    Args:
        label: Short header naming which agent/checkpoint this is for.
        failure_counts: A dict mapping a failure reason to how many times
            it happened, e.g. {"crash": 0, "unsafe": 5, "timeout": 1}.
    """
    print(f"\nFailure breakdown, {label}:")
    if any(failure_counts.values()):
        for reason, count in failure_counts.items():
            if count:
                print(f"  {reason:<16} {count}")
    else:
        print("  N/A")


def mean_pairwise_angle_deg(unit_vectors):
    """Measure how spread out a set of directions is, in degrees.

    Compares every direction against every other and averages the
    angles between them. Near 0 degrees means every episode approached
    from the same line (a memorized corridor); near 90 degrees means
    approaches point all over (each episode keeps its own start).

    Args:
        unit_vectors: A list of 3-element numpy arrays, each already
            scaled to length 1.

    Returns:
        Mean angle in degrees, or 0.0 with fewer than two directions.
    """
    import numpy as np  # local import: keeps this module numpy-free for path-only callers

    if len(unit_vectors) < 2:
        return 0.0
    # dtype=float is significant: an integer input would make dots integer too,
    # and the in-place clip below cannot write float bounds into it.
    vectors = np.array(unit_vectors, dtype=float)
    dots = vectors @ vectors.T  # dot product of unit vectors = cosine of the angle
    np.clip(dots, -1.0, 1.0, out=dots)  # guard against rounding error breaking arccos
    angles = np.degrees(np.arccos(dots))
    # Upper triangle only: each pair once, excluding the 0-degree diagonal.
    iu = np.triu_indices(len(vectors), k=1)
    return float(np.mean(angles[iu]))


def unique_checkpoint_path(run_dir, checkpoint_prefix, model_id, stage, start_epoch):
    """Find a checkpoint path that doesn't already exist on disk.

    Starts at start_epoch for the given stage and counts up.

    Used when a trainer promotes its best-scoring epoch to a new number.
    Checks disk directly rather than trusting arithmetic, so a promoted
    checkpoint can never overwrite a real training epoch.

    Args:
        run_dir: Checkpoint folder to search in.
        checkpoint_prefix: Filename prefix, e.g. "nodrift_curriculum_ppo_model".
        model_id: Run number, matching the folder's run number.
        stage: Stage number (or the single "final" stage for a
            no-curriculum run).
        start_epoch: First epoch number to try.

    Returns:
        A full path (without the ".zip" extension, matching how the
        trainers call model.save()) that does not currently exist on disk.
    """
    epoch = start_epoch
    while True:
        candidate = os.path.join(
            run_dir, f"{checkpoint_prefix}_{model_id}_{stage}_{epoch}"
        )
        if not os.path.exists(candidate + ".zip"):
            return candidate
        epoch += 1


def _json_metadata_default(obj):
    """Fallback encoder for values json.dump can't handle natively.

    Numpy arrays and scalars (e.g. a fixed_state array, a float32 reward
    coefficient) go through .tolist(), giving real JSON numbers/lists
    that later code (or a notebook) can load and use directly, instead
    of an unparseable str() repr like "[100   0   0   0   0   0   0]".
    Anything else without .tolist() still falls back to str(), so a
    metadata write never blocks training on an unexpected type.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def save_run_metadata(run_dir, metadata):
    """Write (or overwrite) run_metadata.json in a checkpoint run folder.

    Call once before training starts (so a record survives even if the
    run crashes or is interrupted) and again after it finishes with
    results merged in. nodrift_train.py also calls this every epoch, so
    the file stays a near-current backup of per-epoch data throughout a
    run, not just a start/end snapshot.

    Args:
        run_dir: Checkpoint folder to write into.
        metadata: JSON-serializable dict of run configuration/results.
            Non-serializable values are passed through
            _json_metadata_default rather than raising, so a metadata
            write never blocks training.

    Returns:
        The path written to.
    """
    path = os.path.join(run_dir, "run_metadata.json")
    # Write to a temp file, then rename, so a crash mid-write leaves the
    # previous complete file in place. os.replace is atomic on Windows
    # and POSIX.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_metadata_default)
        f.write("\n")  # trailing newline avoids a "no newline at end of file" diff warning
    os.replace(tmp_path, path)
    return path


def resolve_env_config_from_metadata(checkpoint_path):
    """Look up the exact environment config a checkpoint trained under.

    Reads the checkpoint's own run_metadata.json instead of a live
    get_curriculum() call, which can return the wrong values if the
    curriculum module has since been edited.

    Handles both metadata shapes: a single "resolved_config" for
    non-curriculum runs, or a "resolved_curriculum" dict keyed by stage
    for curriculum-trained checkpoints.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.

    Returns:
        A (config, source) tuple. config is a dict ready to pass as
        **kwargs to the environment class, or None if no usable metadata
        exists. source is a short string describing where it came from
        (or why it failed), meant for printing.
    """
    run_dir = os.path.dirname(checkpoint_path)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    # Relative to the repo root, so printed source strings say which run
    # the config came from instead of a bare, ambiguous "run_metadata.json".
    try:
        rel_path = os.path.relpath(metadata_path, _REPO_ROOT)
    except ValueError:
        rel_path = metadata_path  # different drive on Windows; can't be relative

    if not os.path.isfile(metadata_path):
        return None, "no run_metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    if "resolved_config" in metadata:
        # Single-run trainer (nodrift_train.py): one config for the run.
        return metadata["resolved_config"], rel_path

    if "resolved_curriculum" in metadata:
        stage = extract_stage(checkpoint_path)
        stage_key = str(stage)
        stage_entry = metadata["resolved_curriculum"].get(stage_key)
        if stage_entry is None:
            available = sorted(metadata["resolved_curriculum"].keys(), key=int)
            return None, f"stage {stage} missing from {rel_path} (has {available})"
        return stage_entry["config"], f"{rel_path}, stage {stage_key}"

    return None, f"{rel_path} has no usable config"


def describe_run_type(checkpoint_path):
    """Label a checkpoint as curriculum or standalone training.

    Best effort: falls back to the folder name when metadata is absent.

    Checks run_metadata.json first ("resolved_curriculum" vs.
    "resolved_config"), falling back to the run folder's naming
    convention for checkpoints with no metadata.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.

    Returns:
        "curriculum" or "no curriculum".
    """
    run_dir = os.path.dirname(checkpoint_path)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        if "resolved_curriculum" in metadata:
            return "curriculum"
        if "resolved_config" in metadata:
            return "no curriculum"
    return "curriculum" if "curriculum" in os.path.basename(run_dir) else "no curriculum"


def load_run_metadata(checkpoint_path):
    """Load a checkpoint's full run_metadata.json, if it has one.

    Unlike resolve_env_config_from_metadata (which extracts just the
    environment/curriculum config for building an eval env), this returns
    the whole metadata dict: seed, env_flags, hyperparameters,
    curriculum_variant, etc. Used by summarize_nodrift_run/
    auto_label_checkpoints to classify what kind of comparison two
    checkpoints represent.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.

    Returns:
        The metadata dict, or None if the run folder has no
        run_metadata.json (checkpoints predating that feature).
    """
    run_dir = os.path.dirname(checkpoint_path)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path) as f:
        return json.load(f)


# Old env-flags key names, before these were renamed off the saferl_
# prefix. normalize_flag_keys() translates them for old checkpoints.
_LEGACY_FLAG_KEY_ALIASES = {
    "custom_braking_margin_obs": "saferl_braking_margin_obs",
    "custom_vel_penalty": "saferl_vel_penalty",
}


def normalize_flag_keys(recorded_flags):
    """Rewrite a checkpoint's env_flags dict to use current key names.

    Without this, comparing an old checkpoint (old key name) against a
    new one reads the old checkpoint's flag as missing and fills it with
    False. That reports a real difference as no difference.

    Args:
        recorded_flags: The raw flat env-flags dict from a
            run_metadata.json, using either old or new key names.

    Returns:
        A new dict with legacy keys renamed. A key already under its
        current name always wins.
    """
    normalized = dict(recorded_flags)
    for current_key, legacy_key in _LEGACY_FLAG_KEY_ALIASES.items():
        if legacy_key in normalized:
            value = normalized.pop(legacy_key)
            normalized.setdefault(current_key, value)
    return normalized


# Which family each env flag belongs to, for run_metadata.json's
# "env_flags" block. Unlisted flags go to "other".
_ENV_FLAG_FAMILIES = {
    "saferl_obs": "saferl",
    "saferl_exp_dist_reward": "saferl",
    "saferl_delta_v_penalty": "saferl",
    "saferl_success_time_bonus": "saferl",
    "saferl_success_reward": "saferl",
    "saferl_vel_constraint": "saferl",
    "saferl_max_distance": "saferl",
    "saferl_no_time_penalty": "saferl",
    "custom_braking_margin_obs": "custom",
    "custom_vel_penalty": "custom",
    "normalize_obs": "custom",
    "space_controls_obs": "space_controls",
    "space_controls_time_penalty": "space_controls",
    "space_controls_init_vel_fraction": "space_controls",
    "space_controls_speed_limit_offset": "space_controls",
    "space_controls_vel_constraint": "space_controls",
    "space_controls_dock_radius": "space_controls",
    "space_controls_max_distance": "space_controls",
    "space_controls_low_thrust": "space_controls",
    "paper_time_penalty": "paper",
    "paper_unsafe_reward": "paper",
}

# Family order in a grouped env_flags block, fixed so every
# run_metadata.json reads the same way regardless of dict insertion order.
_ENV_FLAG_FAMILY_ORDER = ("saferl", "space_controls", "custom", "paper", "other")


def group_env_flags(flat_flags):
    """Group a flat env-flags dict into families for run_metadata.json.

    Splits by flag name using _ENV_FLAG_FAMILIES ("saferl", "custom",
    "space_controls"), with anything unlisted under "other". Purely a
    display/storage grouping: env construction still needs a flat dict,
    so any reader should call ungroup_env_flags() to flatten this back out.

    Args:
        flat_flags: A flat {flag_name: bool} dict, e.g. a trainer's
            ENV_FLAGS constant.

    Returns:
        A dict of family name to {flag_name: bool}, in a fixed family
        order. A family with no flags in flat_flags is left out entirely.
    """
    grouped = {family: {} for family in _ENV_FLAG_FAMILY_ORDER}
    for key, value in flat_flags.items():
        family = _ENV_FLAG_FAMILIES.get(key, "other")
        grouped[family][key] = value
    return {family: flags for family, flags in grouped.items() if flags}


def ungroup_env_flags(maybe_grouped_flags):
    """Flatten a run_metadata.json env-flags block back to one flat dict.

    Accepts either the current grouped shape (see group_env_flags) or an
    older flat dict, so callers do not need to know which shape a given
    checkpoint's metadata used.

    Args:
        maybe_grouped_flags: A family-to-{flag_name: bool} dict, or an
            already-flat {flag_name: bool} dict.

    Returns:
        A flat {flag_name: bool} dict.
    """
    if not maybe_grouped_flags:
        return {}
    first_value = next(iter(maybe_grouped_flags.values()))
    if not isinstance(first_value, dict):
        return dict(maybe_grouped_flags)
    flat = {}
    for family_flags in maybe_grouped_flags.values():
        flat.update(family_flags)
    return flat


def _read_env_flags(metadata):
    """Read and flatten a run_metadata.json's env-flags block.

    Reads the current "env_flags" key (grouped by family) if present,
    falling back to the older flat "saferl_flags" key for checkpoints
    trained before that rename. Legacy per-flag key names are also
    translated (see normalize_flag_keys).

    Args:
        metadata: A loaded run_metadata.json dict, or None.

    Returns:
        A flat {flag_name: bool} dict. Only contains flags the checkpoint
        recorded; empty if metadata is None or has neither key.
        Use _complete_env_flags() to fill in the rest as False.
    """
    if not metadata:
        return {}
    raw = metadata.get("env_flags")
    if raw is None:
        raw = metadata.get("saferl_flags", {})
    return normalize_flag_keys(ungroup_env_flags(raw))


def _complete_env_flags(metadata, known_keys):
    """Read a checkpoint's env_flags, filling in every unrecorded flag as False.

    Every flag in this project defaults to False, and is only ever True
    where a trainer's ENV_FLAGS explicitly turned it on. So a flag
    missing from a checkpoint's metadata is historical fact ("this
    checkpoint trained without it"), not a genuine unknown, and must
    default to False rather than being left out. Leaving it out was a
    real bug here (see resolve_env_flags_from_metadata's docstring): a
    checkpoint comparison could report a flag as "different" just
    because one side never recorded it, even when both sides
    trained with it off.

    Also reports which flags were filled in this way, an "audit trail"
    so a caller can tell a reader when a comparison is trusting an
    assumption (this checkpoint predates the flag, so it must be False)
    rather than a fact the checkpoint's own metadata recorded.

    Args:
        metadata: A loaded run_metadata.json dict, or None.
        known_keys: Every flag name that should be present in the
            result, e.g. _ENV_FLAG_SHORT_NAMES.keys() for nodrift or
            _DRIFT_ENV_FLAG_SHORT_NAMES.keys() for drift.

    Returns:
        A (env_flags, assumed_flags) tuple:
            env_flags: {flag_name: bool}, one entry per key in
                known_keys. True/False exactly as recorded where the
                checkpoint recorded it, False where it didn't.
            assumed_flags: sorted list of the flag names that were not
                recorded and got the default False filled in.
    """
    recorded = _read_env_flags(metadata)
    env_flags = {k: recorded.get(k, False) for k in known_keys}
    assumed_flags = sorted(k for k in known_keys if k not in recorded)
    return env_flags, assumed_flags


# Flag-name sets the eval scripts pass as resolve_env_flags_from_metadata's
# "keys". Named here so every script reads back the same flags and adding a
# flag updates all of them at once.

# Every DRL for Space Controls (2024) flag.
SPACE_CONTROLS_FLAG_KEYS = (
    "space_controls_obs",
    "space_controls_time_penalty",
    "space_controls_init_vel_fraction",
    "space_controls_speed_limit_offset",
    "space_controls_vel_constraint",
    "space_controls_dock_radius",
    "space_controls_max_distance",
    "space_controls_low_thrust",
)

# Reward-side flags that do not change observation length, so they cannot
# be inferred from observation size and must be read from metadata.
NODRIFT_REWARD_FLAG_KEYS = (
    "saferl_exp_dist_reward",
    "saferl_delta_v_penalty",
    "saferl_success_time_bonus",
    "saferl_success_reward",
    "saferl_vel_constraint",
    "normalize_obs",
    # paper_time_penalty changes reward only, so leaving it off still gives
    # the right dock rate and fuel; only the reward column is wrong.
    # docking_env.py has no paper_unsafe_reward counterpart.
    "paper_time_penalty",
    "saferl_no_time_penalty",
    # saferl_max_distance changes termination (max_boundary_box), not just
    # reward: a checkpoint trained at the 40km boundary would see spurious
    # out-of-bounds endings at the default 200m.
    "saferl_max_distance",
)

# Same, plus custom_vel_penalty and paper_unsafe_reward, which only the
# drift environment has.
DRIFT_REWARD_FLAG_KEYS = (
    "saferl_exp_dist_reward",
    "saferl_delta_v_penalty",
    "saferl_success_time_bonus",
    "saferl_success_reward",
    "custom_vel_penalty",
    "saferl_vel_constraint",
    "normalize_obs",
    "paper_time_penalty",
    "paper_unsafe_reward",
    "saferl_no_time_penalty",
    "saferl_max_distance",
)


# Metric definitions, shared so every eval script reports the same thing.
# These four decide what a reported number means, not how a run behaves or
# a figure looks, so every paired-comparison script imports them rather
# than defining its own. Values are also published in README, so treat
# them as fixed.

# Coast fraction: share of steps with L2 thrust below this (of a 1.0 N max).
COAST_THRUST_THRESHOLD = 0.1

# True coast fraction: share of steps below this much tighter threshold,
# where fuel spent is negligible. Separates drift's real zero-thrust glide
# from a nodrift policy merely thrusting gently.
TRUE_COAST_THRESHOLD = 0.01

# Saturation fraction: share of steps with any action axis above this,
# i.e. bang-bang control rather than graded correction.
ACTION_SATURATION_THRESHOLD = 0.99

# Approach direction spread: each episode's approach direction is the unit
# vector from chief to deputy at the first step inside this distance (m).
APPROACH_CHECK_DISTANCE = 10.0

# Per-episode step limit, a safety net rather than a normal ending: it stops
# a bad checkpoint from looping forever. Shared for the same reason as the
# four constants above: a script that hit this limit at a different value
# than another script would classify that episode's win/loss differently.
MAX_STEPS_PER_EPISODE = 5000


def resolve_env_flags_from_metadata(checkpoint_path, keys=None):
    """Read a checkpoint's recorded env_flags back out of its metadata.

    resolve_obs_flags() can infer obs-shaping flags from observation
    size, but reward-side flags (normalize_obs, custom_vel_penalty,
    saferl_vel_constraint, saferl_exp_dist_reward,
    saferl_delta_v_penalty) don't touch the observation, so they must be
    read back from what the trainer recorded.

    A flag not present in a checkpoint's metadata means the checkpoint
    predates that flag, or has no metadata at all (every checkpoint from
    before run_metadata.json existed). Either way that means it trained
    with that flag off: every flag in this project defaults False and
    is only ever True where a trainer's ENV_FLAGS explicitly set it
    "Missing" is historical fact, not
    "unknown," and must never be filled in from an env class's own
    module-level default, since several of those (docking_env.py's
    SAFERL_OBS, SAFERL_EXP_DIST_REWARD, SAFERL_DELTA_V_PENALTY,
    SAFERL_VEL_CONSTRAINT, CUSTOM_BRAKING_MARGIN_OBS, NORMALIZE_OBS) are
    True, which describes what a *new* run should use, not what an old
    checkpoint trained under. Splatting an incomplete flags
    dict into a constructor lets those module defaults leak into the
    eval env with no warning, evaluating the checkpoint under flags it
    never trained with; that was a real bug here.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.
        keys: Optional iterable of flag names to extract (current key
            names). None returns every recorded flag as-is, with no
            defaulting; pass keys to get a complete, safe-to-splat dict.

    Returns:
        A dict keyed by current flag names (old key names are
        translated, see normalize_flag_keys). When keys is given, every
        requested key is present, explicitly False if the checkpoint
        never recorded it, so the result is always safe to splat
        directly into a constructor without relying on that
        constructor's own defaults. When keys is None, returns exactly
        what was recorded, which may be a subset or empty.
    """
    metadata = load_run_metadata(checkpoint_path)
    flags = _read_env_flags(metadata)
    if keys is None:
        return flags
    return {k: flags.get(k, False) for k in keys}


# env_flags keys -> short, print-friendly names, used by
# auto_label_checkpoints when exactly one flag differs.
_ENV_FLAG_SHORT_NAMES = {
    "saferl_obs": "SafeRL obs",
    "saferl_exp_dist_reward": "Exp. distance reward",
    "saferl_delta_v_penalty": "Delta-v penalty",
    "saferl_success_time_bonus": "Success time bonus",
    "saferl_success_reward": "Success reward = +2",
    "saferl_vel_constraint": "Velocity constraint",
    "saferl_max_distance": "40km boundary",
    "saferl_no_time_penalty": "No time penalty",
    "custom_braking_margin_obs": "Braking margin obs (custom)",
    "normalize_obs": "Obs normalization",
    "space_controls_obs": "Space Controls obs (no timestep)",
    "space_controls_time_penalty": "Space Controls flat time penalty",
    "space_controls_init_vel_fraction": "Space Controls 80% init speed limit",
    "space_controls_speed_limit_offset": "Space Controls speed limit offset",
    "space_controls_vel_constraint": "Space Controls velocity constraint",
    "space_controls_dock_radius": "Space Controls 10m dock radius",
    "space_controls_max_distance": "Space Controls 800m boundary",
    "space_controls_low_thrust": "Space Controls 0.1N low thrust",
    "paper_time_penalty": "Paper flat time penalty",
}

# Training-side ablations recorded outside env_flags. Diffed alongside
# env_flags so a sparse or noisy run is classified as a real ablation.
_TRAIN_ABLATION_SHORT_NAMES = {
    "sparse_reward": "Sparse reward",
    "train_action_noise": "Train-time action noise",
}


def _read_train_ablations(metadata):
    """Read the training-side ablation settings out of run metadata.

    Args:
        metadata: A loaded run_metadata.json dict, or None.

    Returns:
        {"sparse_reward": bool, "train_action_noise": False or float}.
        Runs trained before these flags existed default to the pre-flag
        behavior (dense reward, no noise), so they compare equal to a
        run that explicitly recorded those same values.
    """
    if not metadata:
        return {"sparse_reward": False, "train_action_noise": False}
    fraction = metadata.get("train_action_noise_fraction", 0.0) or 0.0
    return {
        "sparse_reward": metadata.get("reward_structure", "dense") == "sparse",
        "train_action_noise": (
            fraction if metadata.get("train_action_noise", False) else False),
    }


def _format_ablation_value(value):
    """Render one flag or ablation value for a comparison label.

    Args:
        value: A bool env-flag value, or a float noise fraction.

    Returns:
        "ON"/"OFF" for booleans, "+/-5%" style for a noise fraction.
    """
    if isinstance(value, bool) or value is None:
        return "ON" if value else "OFF"
    if isinstance(value, (int, float)):
        return f"+/-{value * 100:.0f}%"
    return str(value)


def summarize_nodrift_run(checkpoint_path):
    """Summarize a nodrift checkpoint's training run.

    Normalized so auto_label_checkpoints can diff it against another.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.

    Returns:
        A dict: has_metadata, run_kind ("curriculum"/"standalone"),
        curriculum_variant, seed, env_flags (every known nodrift flag,
        real value where recorded, False where assumed, see
        _complete_env_flags), assumed_flags (which of those were
        assumed rather than recorded), key_hparams (ent_coef/target_kl/
        gamma), run_folder. Fields default to None/{}/False/every-flag
        when the run has no run_metadata.json.
    """
    run_folder = os.path.basename(os.path.dirname(checkpoint_path))
    run_kind = "curriculum" if describe_run_type(checkpoint_path) == "curriculum" else "standalone"
    metadata = load_run_metadata(checkpoint_path)
    env_flags, assumed_flags = _complete_env_flags(metadata, _ENV_FLAG_SHORT_NAMES.keys())
    if metadata is None:
        return {
            "has_metadata": False,
            "run_kind": run_kind,
            "curriculum_variant": None,
            "seed": None,
            "env_flags": env_flags,
            "assumed_flags": assumed_flags,
            "train_ablations": _read_train_ablations(None),
            "key_hparams": {},
            "run_folder": run_folder,
        }
    return {
        "has_metadata": True,
        "run_kind": run_kind,
        "curriculum_variant": metadata.get("curriculum_variant"),
        "seed": metadata.get("seed"),
        "env_flags": env_flags,
        "assumed_flags": assumed_flags,
        "train_ablations": _read_train_ablations(metadata),
        "key_hparams": {
            k: metadata[k] for k in ("ent_coef", "target_kl", "gamma") if k in metadata
        },
        "run_folder": run_folder,
    }


def _describe_assumed_flag_caveat(differing_flags, summary_a, summary_b, short_names):
    """Build a one-line caveat when a reported flag difference was assumed.

    _complete_env_flags() fills in a flag a checkpoint never recorded as
    False, its documented default (see that function's docstring
    for why False is the right default here). That is correct, but it
    is still an assumption about a checkpoint's history, not something
    its own metadata says directly. This tells a reader when that
    happened, for exactly the flag(s) that ended up in the reported
    differences, an "audit trail" so a surprising ablation result can be
    traced back to "checkpoint X never recorded this flag" instead of
    looking unexplained.

    Args:
        differing_flags: {flag_name: (value_a, value_b)}, the flags a
            comparison found different (see auto_label_checkpoints /
            auto_label_drift_checkpoints).
        summary_a: The summarize_*_run() dict for side A.
        summary_b: Same, for side B.
        short_names: The flag_name -> display name lookup to use
            (_ENV_FLAG_SHORT_NAMES or _DRIFT_ENV_FLAG_SHORT_NAMES).

    Returns:
        A caveat sentence, or None if none of the reported differences
        involved a flag that had to be assumed rather than recorded.
    """
    assumed_in_a = set(summary_a.get("assumed_flags", [])) & set(differing_flags)
    assumed_in_b = set(summary_b.get("assumed_flags", [])) & set(differing_flags)
    if not assumed_in_a and not assumed_in_b:
        return None

    def _names(keys):
        return ", ".join(short_names.get(k, k) for k in sorted(keys))

    parts = []
    if assumed_in_a:
        parts.append(f"{summary_a['run_folder']} never recorded "
                     f"{_names(assumed_in_a)} (assumed OFF, the documented default)")
    if assumed_in_b:
        parts.append(f"{summary_b['run_folder']} never recorded "
                     f"{_names(assumed_in_b)} (assumed OFF, the documented default)")
    return "Assumed, not recorded:\n- " + "\n- ".join(parts) + "."


def _append_note(existing_note, extra_note):
    """Join two printable notes, skipping either one that is empty.

    Args:
        existing_note: A note string, or None.
        extra_note: A second note string, or None.

    Returns:
        Both joined with a newline if both are set, whichever one is set
        if only one is, or None if neither is.
    """
    if existing_note and extra_note:
        return f"{existing_note}\n{extra_note}"
    return existing_note or extra_note


def auto_label_checkpoints(path_a, path_b, default_label_a="A", default_label_b="B"):
    """Detect what kind of comparison two nodrift checkpoints represent.

    Generates accurate labels for it, instead of a hand-written
    LABEL_A/LABEL_B that can become outdated with no warning.

    Checks run_kind, curriculum_variant, then env_flags, in that
    order, and returns the first of these that differs:
        "curriculum_vs_standalone"  one is curriculum-trained, one isn't
        "curriculum_variant"        both curriculum, different variant
        "flag_ablation"             exactly one env_flags entry differs
        "multi_flag_ablation"       more than one flag differs
        "hparam_ablation"           flags identical, a hyperparameter differs
        "seed_only"                 everything else identical, only seed differs
        "identical"                 nothing differs, including seed
        "unlabeled"                 one or both have no run_metadata.json

    Args:
        path_a: Full path to checkpoint A.
        path_b: Full path to checkpoint B.
        default_label_a: Fallback label for A when auto-detection can't run.
        default_label_b: Same, for B.

    Returns:
        A dict: label_a, label_b (str); comparison_type (one of the
        strings above); note (printable warning, set for "seed_only"
        and "unlabeled"); differences ({field: (value_a, value_b)} for
        every field that differs).
    """
    summary_a = summarize_nodrift_run(path_a)
    summary_b = summarize_nodrift_run(path_b)

    if not summary_a["has_metadata"] or not summary_b["has_metadata"]:
        missing = []
        if not summary_a["has_metadata"]:
            missing.append(f"A ({summary_a['run_folder']})")
        if not summary_b["has_metadata"]:
            missing.append(f"B ({summary_b['run_folder']})")
        return {
            "label_a": default_label_a,
            "label_b": default_label_b,
            "comparison_type": "unlabeled",
            "note": (
                f"{' and '.join(missing)} have no run_metadata.json, so the "
                f"comparison type could not be auto-detected; using default labels."
            ),
            "differences": {},
        }

    # Build the full differences dict, available regardless of which
    # category ends up being the "primary" one.
    differences = {}
    if summary_a["run_kind"] != summary_b["run_kind"]:
        differences["run_kind"] = (summary_a["run_kind"], summary_b["run_kind"])
    if summary_a["curriculum_variant"] != summary_b["curriculum_variant"]:
        differences["curriculum_variant"] = (
            summary_a["curriculum_variant"], summary_b["curriculum_variant"]
        )
    all_flag_keys = set(summary_a["env_flags"]) | set(summary_b["env_flags"])
    differing_flags = {
        k: (summary_a["env_flags"].get(k), summary_b["env_flags"].get(k))
        for k in sorted(all_flag_keys)
        if summary_a["env_flags"].get(k) != summary_b["env_flags"].get(k)
    }
    # Training-side ablations live outside env_flags but classify the
    # same way, so fold them into the existing flag branches.
    differing_flags.update({
        k: (summary_a["train_ablations"].get(k, False),
            summary_b["train_ablations"].get(k, False))
        for k in _TRAIN_ABLATION_SHORT_NAMES
        if summary_a["train_ablations"].get(k, False)
        != summary_b["train_ablations"].get(k, False)
    })
    differences.update(differing_flags)
    all_hparam_keys = set(summary_a["key_hparams"]) | set(summary_b["key_hparams"])
    differing_hparams = {
        k: (summary_a["key_hparams"].get(k), summary_b["key_hparams"].get(k))
        for k in sorted(all_hparam_keys)
        if summary_a["key_hparams"].get(k) != summary_b["key_hparams"].get(k)
    }
    differences.update(differing_hparams)
    if summary_a["seed"] != summary_b["seed"]:
        differences["seed"] = (summary_a["seed"], summary_b["seed"])

    # Classify by the highest-priority axis that differs.
    if summary_a["run_kind"] != summary_b["run_kind"]:
        def _kind_label(summary):
            if summary["run_kind"] == "curriculum" and summary["curriculum_variant"]:
                return f"Curriculum ({summary['curriculum_variant']})"
            return "Curriculum" if summary["run_kind"] == "curriculum" else "Standalone (no curriculum)"
        return {
            "label_a": _kind_label(summary_a),
            "label_b": _kind_label(summary_b),
            "comparison_type": "curriculum_vs_standalone",
            "note": None,
            "differences": differences,
        }

    if summary_a["curriculum_variant"] != summary_b["curriculum_variant"]:
        return {
            "label_a": f"Curriculum ({summary_a['curriculum_variant']})",
            "label_b": f"Curriculum ({summary_b['curriculum_variant']})",
            "comparison_type": "curriculum_variant",
            "note": None,
            "differences": differences,
        }

    if len(differing_flags) == 1:
        flag_name, (val_a, val_b) = next(iter(differing_flags.items()))
        short_name = _ENV_FLAG_SHORT_NAMES.get(
            flag_name, _TRAIN_ABLATION_SHORT_NAMES.get(flag_name, flag_name))
        return {
            "label_a": f"{short_name} {_format_ablation_value(val_a)}",
            "label_b": f"{short_name} {_format_ablation_value(val_b)}",
            "comparison_type": "flag_ablation",
            "note": _describe_assumed_flag_caveat(
                differing_flags, summary_a, summary_b, _ENV_FLAG_SHORT_NAMES),
            "differences": differences,
        }

    if len(differing_flags) > 1:
        flag_list = ", ".join(
            _ENV_FLAG_SHORT_NAMES.get(k, _TRAIN_ABLATION_SHORT_NAMES.get(k, k))
            for k in differing_flags)
        return {
            "label_a": summary_a["run_folder"],
            "label_b": summary_b["run_folder"],
            "comparison_type": "multi_flag_ablation",
            "note": _append_note(
                f"Multiple flags differ: {flag_list}.\n"
                f"Using run folder names instead of a short label.\n"
                f"See the differences table for details.",
                _describe_assumed_flag_caveat(
                    differing_flags, summary_a, summary_b, _ENV_FLAG_SHORT_NAMES),
            ),
            "differences": differences,
        }

    if differing_hparams:
        hparam_name, (val_a, val_b) = next(iter(differing_hparams.items()))
        return {
            "label_a": f"{hparam_name}={val_a}",
            "label_b": f"{hparam_name}={val_b}",
            "comparison_type": "hparam_ablation",
            "note": None,
            "differences": differences,
        }

    if summary_a["seed"] != summary_b["seed"]:
        return {
            "label_a": f"Seed {summary_a['seed']}",
            "label_b": f"Seed {summary_b['seed']}",
            "comparison_type": "seed_only",
            # No note: _COMPARISON_TYPE_DESCRIPTIONS["seed_only"] covers it.
            "note": None,
            "differences": differences,
        }

    return {
        "label_a": summary_a["run_folder"],
        "label_b": summary_b["run_folder"],
        "comparison_type": "identical",
        "note": (
            "No difference was found between these two checkpoints' "
            "recorded run_kind, curriculum_variant, env_flags, key "
            "hyperparameters, or seed (e.g. they may be two epochs of the "
            "same run)."
        ),
        "differences": differences,
    }


# drift_env.py's env_flags keys -> short, print-friendly names, used
# by auto_label_drift_checkpoints when exactly one flag differs.
_DRIFT_ENV_FLAG_SHORT_NAMES = {
    "saferl_obs": "SafeRL obs",
    "saferl_exp_dist_reward": "Exp. distance reward",
    "saferl_delta_v_penalty": "Delta-v penalty",
    "saferl_success_time_bonus": "Success time bonus",
    "saferl_success_reward": "Success reward = +2",
    "custom_braking_margin_obs": "Braking margin obs (custom)",
    "custom_vel_penalty": "Velocity penalty (custom)",
    "saferl_vel_constraint": "Velocity constraint",
    "saferl_max_distance": "40km boundary",
    "saferl_no_time_penalty": "No time penalty",
    "normalize_obs": "Obs normalization",
    "space_controls_obs": "Space Controls obs (no timestep)",
    "space_controls_time_penalty": "Space Controls flat time penalty",
    "space_controls_init_vel_fraction": "Space Controls 80% init speed limit",
    "space_controls_speed_limit_offset": "Space Controls speed limit offset",
    "space_controls_vel_constraint": "Space Controls velocity constraint",
    "space_controls_dock_radius": "Space Controls 10m dock radius",
    "space_controls_max_distance": "Space Controls 800m boundary",
    "space_controls_low_thrust": "Space Controls 0.1N low thrust",
    "paper_time_penalty": "Paper flat time penalty",
    "paper_unsafe_reward": "Paper unsafe reward = 0",
}


def summarize_drift_run(checkpoint_path):
    """Summarize a drift checkpoint's training run.

    Normalized so auto_label_drift_checkpoints can diff it against
    another.

    Args:
        checkpoint_path: Full path to any one checkpoint from the chain
            (run_metadata.json is shared by every stage in the run).

    Returns:
        A dict: has_metadata, seed, env_flags (every known flag; a
        flag missing from the recording defaults to False, its
        documented default, rather than None, so an old checkpoint and
        one that explicitly recorded False compare as equal, see
        _complete_env_flags), assumed_flags (which of those were
        assumed rather than recorded), run_folder.
    """
    run_folder = os.path.basename(os.path.dirname(checkpoint_path))
    metadata = load_run_metadata(checkpoint_path)
    env_flags, assumed_flags = _complete_env_flags(metadata, _DRIFT_ENV_FLAG_SHORT_NAMES.keys())
    if metadata is None:
        return {
            "has_metadata": False,
            "seed": None,
            "env_flags": env_flags,
            "assumed_flags": assumed_flags,
            "train_ablations": _read_train_ablations(None),
            "run_folder": run_folder,
        }
    return {
        "has_metadata": True,
        "seed": metadata.get("seed"),
        "env_flags": env_flags,
        "assumed_flags": assumed_flags,
        "train_ablations": _read_train_ablations(metadata),
        "run_folder": run_folder,
    }


def auto_label_drift_checkpoints(path_a, path_b, default_label_a="A", default_label_b="B"):
    """Detect what kind of comparison two drift chains represent.

    Generates accurate labels for it. Drift counterpart to
    auto_label_checkpoints (nodrift).

    Simpler than the nodrift version: drift has only one curriculum
    shape, so there's no run_kind/curriculum_variant axis, only the
    env_flags and the seed. Classifies as the first that differs:
        "flag_ablation"        exactly one env_flags entry differs
        "multi_flag_ablation"  more than one flag differs
        "seed_only"            every flag identical, only the seed differs
        "identical"            nothing differs, including seed
        "unlabeled"            one or both chains have no run_metadata.json

    Args:
        path_a: Full path to any one checkpoint from chain A.
        path_b: Full path to any one checkpoint from chain B.
        default_label_a: Label to use for A when auto-detection can't run
            (no metadata for one or both chains).
        default_label_b: Same, for B.

    Returns:
        A dict: label_a, label_b (str), comparison_type (str), note (str
        or None), differences (dict of {field: (value_a, value_b)}).
    """
    summary_a = summarize_drift_run(path_a)
    summary_b = summarize_drift_run(path_b)

    if not summary_a["has_metadata"] or not summary_b["has_metadata"]:
        missing = []
        if not summary_a["has_metadata"]:
            missing.append(f"A ({summary_a['run_folder']})")
        if not summary_b["has_metadata"]:
            missing.append(f"B ({summary_b['run_folder']})")
        return {
            "label_a": default_label_a,
            "label_b": default_label_b,
            "comparison_type": "unlabeled",
            "note": (
                f"{' and '.join(missing)} have no run_metadata.json, so the "
                f"comparison type could not be auto-detected; using default labels."
            ),
            "differences": {},
        }

    differences = {}
    all_flag_keys = set(summary_a["env_flags"]) | set(summary_b["env_flags"])
    differing_flags = {
        k: (summary_a["env_flags"].get(k), summary_b["env_flags"].get(k))
        for k in sorted(all_flag_keys)
        if summary_a["env_flags"].get(k) != summary_b["env_flags"].get(k)
    }
    # Training-side ablations live outside env_flags but classify the
    # same way, so fold them into the existing flag branches.
    differing_flags.update({
        k: (summary_a["train_ablations"].get(k, False),
            summary_b["train_ablations"].get(k, False))
        for k in _TRAIN_ABLATION_SHORT_NAMES
        if summary_a["train_ablations"].get(k, False)
        != summary_b["train_ablations"].get(k, False)
    })
    differences.update(differing_flags)
    if summary_a["seed"] != summary_b["seed"]:
        differences["seed"] = (summary_a["seed"], summary_b["seed"])

    if len(differing_flags) == 1:
        flag_name, (val_a, val_b) = next(iter(differing_flags.items()))
        short_name = _DRIFT_ENV_FLAG_SHORT_NAMES.get(
            flag_name, _TRAIN_ABLATION_SHORT_NAMES.get(flag_name, flag_name))
        return {
            "label_a": f"{short_name} {_format_ablation_value(val_a)}",
            "label_b": f"{short_name} {_format_ablation_value(val_b)}",
            "comparison_type": "flag_ablation",
            "note": _describe_assumed_flag_caveat(
                differing_flags, summary_a, summary_b, _DRIFT_ENV_FLAG_SHORT_NAMES),
            "differences": differences,
        }

    if len(differing_flags) > 1:
        flag_list = ", ".join(
            _DRIFT_ENV_FLAG_SHORT_NAMES.get(
                k, _TRAIN_ABLATION_SHORT_NAMES.get(k, k))
            for k in differing_flags)
        return {
            "label_a": summary_a["run_folder"],
            "label_b": summary_b["run_folder"],
            "comparison_type": "multi_flag_ablation",
            "note": _append_note(
                f"Multiple flags differ: {flag_list}.\n"
                f"Using run folder names instead of a short label.\n"
                f"See the differences table for details.",
                _describe_assumed_flag_caveat(
                    differing_flags, summary_a, summary_b, _DRIFT_ENV_FLAG_SHORT_NAMES),
            ),
            "differences": differences,
        }

    if summary_a["seed"] != summary_b["seed"]:
        return {
            "label_a": f"Seed {summary_a['seed']}",
            "label_b": f"Seed {summary_b['seed']}",
            "comparison_type": "seed_only",
            # No note: _COMPARISON_TYPE_DESCRIPTIONS["seed_only"] covers it.
            "note": None,
            "differences": differences,
        }

    return {
        "label_a": summary_a["run_folder"],
        "label_b": summary_b["run_folder"],
        "comparison_type": "identical",
        "note": (
            "No difference was found between these two checkpoint chains' "
            "recorded env_flags or seed (e.g. they may be the same run)."
        ),
        "differences": differences,
    }


# Combined drift + nodrift flag-name lookup, so describe_comparison_type
# can name the exact flag that differs regardless of which side called it.
_ALL_FLAG_DISPLAY_NAMES = {
    **_ENV_FLAG_SHORT_NAMES,
    **_DRIFT_ENV_FLAG_SHORT_NAMES,
    **_TRAIN_ABLATION_SHORT_NAMES,
}

# Plain-English explanation for each comparison_type value the two
# auto_label_* functions can return.
_COMPARISON_TYPE_DESCRIPTIONS = {
    "flag_ablation": [
        "One training setting differs between A and B: {flags}.",
        "Everything else about how they were trained is the same.",
        "A difference in the results below is consistent with that "
        "setting, but this is one seed per condition, so seed variance "
        "alone has not been ruled out.",
    ],
    "multi_flag_ablation": [
        "More than one training setting differs between A and B: {flags}.",
        "Any difference in the results below could be caused by any one "
        "of those settings, or by the combination of them.",
        "This is not a clean test of a single setting.",
    ],
    "hparam_ablation": [
        "A training hyperparameter differs between A and B, not a SafeRL "
        "on/off setting.",
        "Any difference in the results below could be caused by that "
        "hyperparameter change.",
    ],
    "curriculum_vs_standalone": [
        "One agent trained step by step with a curriculum, starting easy "
        "and gradually getting harder.",
        "The other trained directly at full difficulty, with no curriculum.",
        "Any difference below could be caused by that training approach, "
        "not by a training setting.",
    ],
    "curriculum_variant": [
        "Both agents trained with a curriculum, but the two curriculums "
        "were not the same shape.",
        "Any difference below could be caused by that difference, not by "
        "a training setting.",
    ],
    "seed_only": [
        "A and B trained with the exact same settings.",
        "The only difference is the random seed (the starting number used "
        "to generate random choices during training).",
    ],
    "identical": [
        "No difference was found between A and B's recorded settings or seed.",
        "They may be the same run evaluated twice, or duplicates.",
    ],
    "unlabeled": [
        "We could not tell what differs, because one or both checkpoints "
        "are missing their saved settings file (run_metadata.json).",
    ],
}


def describe_comparison_type(comparison_type, differences=None):
    """Explain a comparison_type code in plain English.

    Turns a code such as "flag_ablation" into short sentences saying
    what that comparison means.

    The comparison_type code (e.g. "flag_ablation") is meant for programs
    to check, not for a reader to decode, so print these sentences
    instead of (or next to) the raw code.

    Args:
        comparison_type: A "comparison_type" value from
            auto_label_checkpoints or auto_label_drift_checkpoints.
        differences: Optional "differences" dict from those functions.
            For flag_ablation/multi_flag_ablation, names the exact
            setting(s) that differ instead of speaking generically.

    Returns:
        A list of short sentences. Falls back to a generic sentence for
        an unrecognized comparison_type.
    """
    lines = _COMPARISON_TYPE_DESCRIPTIONS.get(
        comparison_type,
        ["This comparison type is not recognized; treat the differences "
         "list below as the source of truth for what changed."],
    )

    if comparison_type in ("flag_ablation", "multi_flag_ablation"):
        differences = differences or {}
        flag_keys = [k for k in differences if k in _ALL_FLAG_DISPLAY_NAMES]
        flags = ", ".join(_ALL_FLAG_DISPLAY_NAMES[k] for k in flag_keys) or "a training setting"
        lines = [line.format(flags=flags) if "{flags}" in line else line for line in lines]

    return lines


def describe_drift_chain(model_paths):
    """Describe a drift checkpoint chain in one short line.

    For printed tables and plot titles, e.g. "PPO run 25, stage 8 to 0"
    instead of raw chained filenames.

    Args:
        model_paths: Checkpoint paths for one chain, hardest-stage-first
            (see resolve_drift_model_chain). At least one entry.

    Returns:
        A short string. If the run number can't be parsed, only the
        stage range is shown.
    """
    first_path = str(model_paths[0])
    last_path = str(model_paths[-1])
    run_number = extract_run(first_path)
    first_stage = extract_stage(first_path)
    last_stage = extract_stage(last_path)

    stage_range = f"stage {first_stage} to {last_stage}"
    if run_number == -1:
        return stage_range
    return f"PPO run {run_number}, {stage_range}"


# nodrift's curriculum was renumbered at stage 4+, so a live
# get_curriculum() lookup is only safe at stages 0-3.
NODRIFT_SAFE_LIVE_LOOKUP_MAX_STAGE = 3


def resolve_env_config_best_effort(checkpoint_path, fallback_config=None):
    """Resolve the best available environment config for a checkpoint.

    Tries, in order:
        1. The checkpoint's own run_metadata.json (exact).
        2. A live get_curriculum() lookup, only where safe: any drift
           checkpoint, or a nodrift checkpoint at stage 0-3 (see
           NODRIFT_SAFE_LIVE_LOOKUP_MAX_STAGE).
        3. fallback_config, if given, marked unverified. Last resort for
           nodrift stage 4+ with no metadata, where a live lookup could
           return a wrong-but-plausible config with no error.

    Args:
        checkpoint_path: Full path to a checkpoint .zip file.
        fallback_config: Config dict to fall back to, unverified. None
            to get (None, ...) back if steps 1-2 can't resolve anything.

    Returns:
        A (config, source, verified) tuple. verified is True only for
        steps 1-2; False for the fallback or an unresolved checkpoint.
    """
    config, source = resolve_env_config_from_metadata(checkpoint_path)
    if config is not None:
        return config, source, True

    mode = detect_agent_mode(checkpoint_path)
    stage = extract_stage(checkpoint_path)

    if mode == "drift" and stage != -1:
        from drift_initial_trainer import get_curriculum as _drift_get_curriculum
        from drift_initial_trainer import NUM_STAGES as _DRIFT_NUM_STAGES
        if stage < _DRIFT_NUM_STAGES:
            drift_config, _ = _drift_get_curriculum(stage)
            return drift_config, f"live curriculum, stage {stage}", True

    if (mode == "nodrift" and stage != -1
            and stage <= NODRIFT_SAFE_LIVE_LOOKUP_MAX_STAGE):
        from nodrift_initial_trainer import get_curriculum as _nodrift_get_curriculum
        from nodrift_initial_trainer import CURRICULUM as _nodrift_curriculum_variant
        # Metadata-less checkpoints all trained under "full". Fall
        # through to the fallback if the trainer variant differs now.
        if _nodrift_curriculum_variant == "full":
            nodrift_config, _ = _nodrift_get_curriculum(stage)
            return nodrift_config, f"live curriculum, stage {stage}", True

    if fallback_config is not None:
        stage_desc = f"stage {stage}" if stage != -1 else "unknown stage"
        return dict(fallback_config), (
            f"no saved training record for this checkpoint ({stage_desc}); "
            f"assuming typical full-range config"
        ), False

    return None, "no metadata, no safe lookup, no fallback given", False


def resolve_drift_stage_thresholds(checkpoint_path, verbose=True):
    """Look up the docking target one DRIFT checkpoint was trained against.

    Every drift curriculum stage docks at a different tightness: an early
    stage counts "docked" from 100 m away, the final stage needs 0.5 m. To
    judge a checkpoint fairly, the evaluation environment has to use that
    checkpoint's own numbers, so this reads them back per checkpoint
    instead of applying one target to the whole chain.

    Resolution order comes from resolve_env_config_best_effort(): the
    checkpoint's own run_metadata.json first (exact), then a live
    get_curriculum() lookup by stage number (safe for drift stages).

    Args:
        checkpoint_path: Full path to one checkpoint .zip file.
        verbose: True (default) prints a note whenever the config could
            not be verified against saved metadata, so you always know
            when a guessed config is in use. Pass False to stay quiet.

    Returns:
        A (pos_thresh, speed_thresh) tuple: how close in meters, and how
        slow in meters per second, the deputy must be to count as docked.
    """
    # Local import avoids a circular import: drift_initial_trainer.py
    # imports this module at its top.
    from drift_initial_trainer import get_curriculum

    stage = extract_stage(checkpoint_path)
    fallback_config = get_curriculum(stage)[0] if stage != -1 else None
    config, source, verified = resolve_env_config_best_effort(
        checkpoint_path, fallback_config=fallback_config
    )
    if verbose and not verified:
        print(f"  Env config: {source}")
    return config["pos_thresh"], config["speed_thresh"]


def resolve_drift_stage_obs_time_norm(checkpoint_path, verbose=False):
    """Look up the max_episode_len one DRIFT checkpoint trained under.

    The twin of resolve_drift_stage_thresholds(), for the observation's
    timestep divisor rather than the docking target. During training a
    drift stage runs 5 to 10 steps, so obs[6] swept 0 to 1 across an
    episode. Chain evaluation runs the same networks for hundreds of
    steps under stage 9's max_episode_len of 9,000, which would pin
    obs[6] near zero throughout. Feeding each stage its own trained
    value restores the input range the policy learned on.

    Same resolution order as resolve_drift_stage_thresholds():
    run_metadata.json first, then a live get_curriculum() lookup, which
    is safe for drift because drift's curriculum has never been
    renumbered.

    Args:
        checkpoint_path: Full path to one checkpoint .zip file.
        verbose: True prints a note when the config could not be
            verified against saved metadata. Defaults False, since the
            caller has usually already printed that same note via
            resolve_drift_stage_thresholds() for this checkpoint.

    Returns:
        The stage's max_episode_len as a float, or None if it could not
        be resolved (caller should then leave obs_time_norm alone).
    """
    # Local import avoids a circular import: drift_initial_trainer.py
    # imports this module at its top.
    from drift_initial_trainer import get_curriculum

    stage = extract_stage(checkpoint_path)
    fallback_config = get_curriculum(stage)[0] if stage != -1 else None
    config, source, verified = resolve_env_config_best_effort(
        checkpoint_path, fallback_config=fallback_config
    )
    if verbose and not verified:
        print(f"  Env config: {source}")
    if config is None:
        return None
    value = config.get("max_episode_len")
    return float(value) if value else None


def resolve_drift_eval_config(overrides=None, verbose=True):
    """Resolve the drift evaluation environment's config (get_curriculum(9)),
    the one config every drift eval script builds its test env from.

    Stage 9 is not a trained stage, so unlike stages 0-8 it is never
    recorded in any checkpoint's run_metadata.json (see
    resolve_env_config_best_effort(), which only ever resolves per-
    checkpoint stage configs). Editing get_curriculum(9)'s stage-9 block
    therefore moves every past drift eval result's interpretation
    retroactively, with no historical record that it happened. This
    function is the one place that lookup happens, so a caller can print
    the resolved values prominently at startup and embed them into saved
    output (see eval_config_export_fields()), instead of the config
    existing only implicitly in whatever get_curriculum(9) happened to
    return on the day a result was produced.

    Args:
        overrides: Optional dict merged over get_curriculum(9)'s own
            config, for a caller-specific ablation (e.g.
            drift_vs_nodrift_test.py's DRIFT_EVAL_LOOKAHEAD). None
            (default) applies no override.
        verbose: True (default) prints the resolved config's key fields.

    Returns:
        A (config, threshold) tuple, same shape as get_curriculum()'s own
        return value, with any overrides applied.
    """
    # Local import avoids a circular import: drift_initial_trainer.py
    # imports this module at its top.
    from drift_initial_trainer import get_curriculum

    config, threshold = get_curriculum(9)
    config = dict(config)
    if overrides:
        config.update(overrides)
    if verbose:
        print(
            f"Drift eval config (get_curriculum(9)): "
            f"max_episode_len={config['max_episode_len']}, "
            f"max_lookahead_len={config['max_lookahead_len']}, "
            f"drift_step_len={config['drift_step_len']}, "
            f"max_boundary_box={config['max_boundary_box']}"
        )
    return config, threshold


def eval_config_export_fields(config, start_state_provenance=None):
    """Pick the scalar fields worth embedding in a saved CSV/JSON result.

    Stage 9's config (see resolve_drift_eval_config()) is never recorded
    in any checkpoint's metadata, so without this a past result's exact
    eval config is only reconstructable by knowing what get_curriculum(9)
    returned on the day that result was produced. Embedding these fields
    directly in the saved output makes a result self-describing even
    after get_curriculum(9)'s stage-9 block is edited later.

    Args:
        config: A drift eval config dict, e.g. from
            resolve_drift_eval_config().
        start_state_provenance: Optional provenance dict from
            resolve_start_states() (its second return value). Bounds like
            eval_min_init_pos_bound/eval_max_init_vel_bound above are
            resolve_drift_eval_config()'s own defaults, which
            resolve_start_states() only samples under when
            given no fixed test set; with a test set loaded, the real
            start states come from whatever bounds that file was
            generated at instead, and the config's bounds are not what
            ran. Passing the provenance here records what the states
            were (a test set's name and fingerprint, or a
            fresh sample's seed), so a saved CSV cannot claim a fresh
            episode was sampled at a bound it was not.  None (the
            default) omits these fields, unchanged from before this
            parameter existed.

    Returns:
        A flat dict of the scalar fields worth keeping (prefixed
        "eval_" so they cannot collide with a result dict's own keys),
        ready to splat into extra_fields= (export_episode_results_csv,
        export_trajectory_steps_csv) or a saved-results dict.
    """
    keys = ("max_episode_len", "max_lookahead_len", "drift_step_len",
           "max_boundary_box", "min_init_pos_bound", "max_init_pos_bound",
           "max_init_vel_bound")
    fields = {f"eval_{k}": config[k] for k in keys if k in config}
    if start_state_provenance is not None:
        p = start_state_provenance
        fields["start_states_source"] = p.get("source")
        if p.get("source") == "test_set":
            fields["start_states_name"] = p.get("name")
            fields["start_states_fingerprint"] = p.get("fingerprint")
        elif p.get("source") == "sampled":
            fields["start_states_seed"] = p.get("seed")
            fields["start_states_fingerprint"] = p.get("fingerprint")
    return fields


def resolve_nodrift_env_config(model_path, fallback_config, verbose=True):
    """Build the environment settings for evaluating a NODRIFT checkpoint.

    Wraps resolve_env_config_best_effort() and drops one field so the
    result can be passed straight into SpaceCraftDockingEnv3D.

    Args:
        model_path: Full path to the checkpoint .zip file.
        fallback_config: Settings dict to fall back on with no saved
            run_metadata.json.
        verbose: True (default) prints which source the settings came from.

    Returns:
        A settings dict ready to splat into SpaceCraftDockingEnv3D(...).
    """
    config, source, verified = resolve_env_config_best_effort(
        model_path, fallback_config=fallback_config
    )
    if verbose:
        print(f"  Env config: {source}")
    # fixed_state is stored as a string in JSON, so drop it: every
    # curriculum stage trains with fixed_start=False anyway.
    config.pop("fixed_state", None)
    return config


def apply_flag_config_overrides(configs, flags, agent_mode):
    """Apply the REPLACE-style flag values to a config dict.

    SPACE_CONTROLS_DOCK_RADIUS / MAX_DISTANCE / LOW_THRUST replace
    pos_thresh / max_boundary_box / max_control inside an environment's
    __init__. Trainers call this before building run_metadata.json
    so the recorded config is what training used, not the
    curriculum's pre-override value. Safe to call more than once: the env
    applies the same override again from the flag, to the same value.

    Was defined separately in docking_env.py and drift_env.py with a
    byte-identical body; moved here once that was confirmed, so a future
    edit only has to happen in one place.

    Args:
        configs: A curriculum config dict. Not modified in place.
        flags: The trainer's ENV_FLAGS dict.
        agent_mode: "drift" or "nodrift", selects which module's
            override values apply (kept in sync with each other, but
            read from the real source rather than duplicated here).

    Returns:
        A new config dict with any active override applied.
    """
    if agent_mode == "drift":
        from drift_env import (
            SPACE_CONTROLS_DOCK_RADIUS_VALUE,
            SPACE_CONTROLS_MAX_DISTANCE_VALUE,
            SPACE_CONTROLS_LOW_THRUST_VALUE,
        )
    else:
        from docking_env import (
            SPACE_CONTROLS_DOCK_RADIUS_VALUE,
            SPACE_CONTROLS_MAX_DISTANCE_VALUE,
            SPACE_CONTROLS_LOW_THRUST_VALUE,
        )

    resolved = dict(configs)
    if flags.get("space_controls_dock_radius"):
        resolved["pos_thresh"] = SPACE_CONTROLS_DOCK_RADIUS_VALUE
    if flags.get("space_controls_max_distance"):
        resolved["max_boundary_box"] = SPACE_CONTROLS_MAX_DISTANCE_VALUE
    if flags.get("space_controls_low_thrust"):
        resolved["max_control"] = SPACE_CONTROLS_LOW_THRUST_VALUE
    return resolved


# Cache of exact discrete-time CWH matrices, keyed by (n, mass, dt).
# Built once per key, since expm() is far too slow to call every step.
_PROPAGATION_MATRIX_CACHE = {}


def propagation_matrices(n, mass, dt):
    """Exact discrete-time CWH matrices for constant thrust over dt.

    Solves the linear system in closed form: state_next = Phi @ state +
    Gamma @ action. Uses the augmented-matrix form of the affine
    solution, since CWH's A is singular and cannot be inverted directly.

    Was defined separately in docking_env.py and drift_env.py with a
    byte-identical body; moved here once that was confirmed. Both
    environments' FAST_ANALYTIC_PROPAGATION flag calls this.

    Args:
        n: Mean motion, rad/s.
        mass: Deputy mass, kg.
        dt: Step duration, seconds.

    Returns:
        A (Phi, Gamma) tuple: the 6x6 state transition matrix and the
        6x3 input matrix.
    """
    key = (n, mass, dt)
    if key not in _PROPAGATION_MATRIX_CACHE:
        import numpy as np  # local import: keeps this module lightweight for path-only callers
        from scipy.linalg import expm

        state_matrix = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -(n ** 2), 0, 0, 0],
        ])
        input_matrix = np.zeros((6, 3))
        input_matrix[3, 0] = input_matrix[4, 1] = input_matrix[5, 2] = 1.0 / mass

        # [[A, B], [0, 0]] exponentiated gives [[Phi, Gamma], [0, I]].
        augmented = np.zeros((9, 9))
        augmented[:6, :6] = state_matrix
        augmented[:6, 6:] = input_matrix
        exponential = expm(augmented * dt)
        _PROPAGATION_MATRIX_CACHE[key] = (exponential[:6, :6], exponential[:6, 6:])
    return _PROPAGATION_MATRIX_CACHE[key]


def apply_dock_radius_override(pos_thresh, flags, agent_mode):
    """Apply the space_controls_dock_radius REPLACE override to a threshold.

    resolve_drift_stage_thresholds()/resolve_nodrift_env_config() read
    pos_thresh straight from a checkpoint's recorded config, which is
    the value BEFORE this flag replaces it (apply_flag_config_overrides()
    above does the same replacement, but only inside the env
    constructor, not in the recorded config). Checkpoints trained before
    that fix was wired into metadata-writing can even have this flag on
    with a recorded
    pos_thresh that never matches what their eval env uses.
    This gets the two back in sync so a threshold comparison is honest.

    Args:
        pos_thresh: The checkpoint's recorded docking distance, meters.
        flags: Flag dict from resolve_env_flags_from_metadata(); only
            "space_controls_dock_radius" is read.
        agent_mode: "drift" or "nodrift", selects which module's
            SPACE_CONTROLS_DOCK_RADIUS_VALUE applies (kept in sync with
            each other, but read from the real source, not duplicated
            here as a second constant that could drift out of sync).

    Returns:
        The effective pos_thresh the checkpoint's eval env will
        use.
    """
    if not flags.get("space_controls_dock_radius"):
        return pos_thresh
    if agent_mode == "drift":
        from drift_env import SPACE_CONTROLS_DOCK_RADIUS_VALUE
    else:
        from docking_env import SPACE_CONTROLS_DOCK_RADIUS_VALUE
    return SPACE_CONTROLS_DOCK_RADIUS_VALUE


def override_dock_threshold(pos_thresh, speed_thresh, pos_override=None,
                            speed_override=None, label=None, verbose=True):
    """Force a checkpoint's win condition to a value it did not train with.

    A checkpoint is normally evaluated against the docking distance/speed
    it trained with (see resolve_drift_stage_thresholds(),
    resolve_nodrift_env_config()). This overrides that after the fact, so
    the same unmodified policy network can be scored against a different
    win condition, e.g. running a checkpoint trained at a 10 m Space
    Controls dock radius against drift's 0.5 m target instead, or the
    reverse (a looser target on a checkpoint trained tight). Useful both
    for a fair head-to-head comparison and as a robustness check: a
    stricter target than training shows whether the policy can still
    close to that precision, not just whatever it was scored on.

    For a DRIFT model chain, apply this only to the final (tightest)
    stage's threshold, e.g. thresholds[-1], not to every stage. The
    non-final stages' thresholds are stage-advance triggers, not the
    episode's actual win condition, and overriding those would change
    when the chain hands off models rather than what counts as docked.

    Args:
        pos_thresh: The checkpoint's own trained docking distance, meters.
        speed_thresh: The checkpoint's own trained docking speed, m/s.
        pos_override: Distance to require instead, meters. None: use
            pos_thresh unchanged.
        speed_override: Speed to require instead, m/s. None: use
            speed_thresh unchanged.
        label: Name to print in the override note, e.g. "Drift" or
            "Nodrift". Required if either override is not None.
        verbose: True prints a note when an override is active.

    Returns:
        An (pos_thresh, speed_thresh) tuple with any overrides applied.
    """
    effective_pos = pos_thresh if pos_override is None else pos_override
    effective_speed = speed_thresh if speed_override is None else speed_override
    if verbose and (pos_override is not None or speed_override is not None):
        print(f"  {label} dock override: {effective_pos:g} m / {effective_speed:g} m/s "
              f"(trained at {pos_thresh:g} m / {speed_thresh:g} m/s)")
    return effective_pos, effective_speed


def warn_if_win_condition_differs(label_a, pos_thresh_a, speed_thresh_a,
                                  label_b, pos_thresh_b, speed_thresh_b):
    """Warn when two paired-comparison sides don't share a win condition.

    A win-rate comparison (and anything downstream of it: the paired
    win-rate test, "Top failure", etc.) is only meaningful if both sides
    have to reach the same distance and speed to count as docked. This
    has gone wrong undetected before: nodrift_PPO_28 (a Space Controls
    replication run) dock at 10 m while its drift comparison partner
    docked at 0.5 m, a 20x looser win condition, with nothing in the
    output flagging it.

    Args:
        label_a: Display name for side A, used in the warning text.
        pos_thresh_a: Side A's effective docking distance, meters
            (apply_dock_radius_override()'d, not the raw recorded value).
        speed_thresh_a: Side A's docking speed threshold, m/s.
        label_b: Display name for side B.
        pos_thresh_b: Side B's effective docking distance, meters.
        speed_thresh_b: Side B's docking speed threshold, m/s.

    Returns:
        True if the two sides match, False if a warning was printed.
    """
    mismatches = []
    if abs(pos_thresh_a - pos_thresh_b) > 1e-9:
        mismatches.append(
            f"dock distance: {label_a} {pos_thresh_a:g} m vs {label_b} {pos_thresh_b:g} m")
    if abs(speed_thresh_a - speed_thresh_b) > 1e-9:
        mismatches.append(
            f"dock speed: {label_a} {speed_thresh_a:g} m/s vs {label_b} {speed_thresh_b:g} m/s")
    if not mismatches:
        return True

    print(f"\nWARNING: {label_a} and {label_b} do not share the same win condition:")
    for mismatch in mismatches:
        print(f"  {mismatch}")
    print("  Win rate and any other win-dependent metric below are not a fair comparison.")
    return False


def resolve_drift_model_chain(checkpoint_spec, checkpoint_root, num_models=100):
    """Find every stage checkpoint of one DRIFT training run, hardest first.

    One training run produces a separate model per curriculum stage.
    Evaluation replays them as a chain: start hardest (100-150 m), hand
    off to the next easier stage on each dock, down to the 0.5 m final dock.

    Args:
        checkpoint_spec: Which run to load, any form resolve_checkpoint_paths()
            accepts: "latest", "run:safe_PPO_26", a glob pattern, or a path.
        checkpoint_root: Path to the data/checkpoints folder.
        num_models: Upper limit on stage checkpoints collected. Keep the
            default (100) high; drift stage numbers track starting
            distance, not precision, so a lower limit could drop the
            tight final-approach stages.

    Returns:
        A list of pathlib.Path objects, hardest stage first.
    """
    from pathlib import Path

    paths = resolve_checkpoint_paths(
        checkpoint_spec, str(checkpoint_root), num_models=num_models,
        folder_prefix="safe_PPO",  # scopes "latest" to drift runs only
    )
    return [Path(p) for p in reversed(paths)]


def build_thrust_colormap(colormap_name, max_thrust):
    """Pick the colors used to shade a trajectory by how hard it thrusted.

    Returns the two matplotlib objects a thrust-colored plot needs: a
    colormap (which colors) and a norm (how to map a thrust value onto them).

    Args:
        colormap_name: "discrete" (default, grey to dark blue in clean
            0.25 N bands, colorblind-safe), "cividis"/"viridis" (smooth,
            colorblind-safe), or "grey_to_warm" (high contrast, NOT
            colorblind-safe, kept for reproducing older figures).
        max_thrust: Largest thrust magnitude in the data, in Newtons.
            Sets the top of the color scale.

    Returns:
        A (colormap, norm) tuple to hand to matplotlib.

    Raises:
        ValueError: If colormap_name is not one of the four names above.
    """
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    if colormap_name == "discrete":
        return build_discrete_thrust_colormap(max_thrust)

    if colormap_name in ("cividis", "viridis"):
        # gamma < 1 stretches the low-to-mid range so small thrusts still show.
        return colormap_name, PowerNorm(gamma=0.5, vmin=0, vmax=max_thrust)

    if colormap_name == "grey_to_warm":
        cmap = LinearSegmentedColormap.from_list(
            "grey_to_warm", ["#1a1a1a", "#39FF14", "#FF2CD4"],
        )
        return cmap, PowerNorm(gamma=0.5, vmin=0, vmax=max_thrust)

    raise ValueError(
        f"Unsupported thrust colormap: {colormap_name!r}. "
        'Use "discrete", "cividis", "viridis", or "grey_to_warm".'
    )


def max_thrust_across_results(*results):
    """Find the single largest thrust value across one or more result sets.

    Lets every line in one figure share a color scale, so the same shade
    always means the same thrust.

    Args:
        *results: One or more result dicts, each with a "thrusts" key
            holding per-episode thrust traces.

    Returns:
        Largest thrust magnitude found, as a float. Falls back to 1.0 if
        there's no data, so callers never divide by zero.
    """
    all_traces = [t for r in results for t in r["thrusts"] if len(t) > 0]
    return max((max(t) for t in all_traces), default=1.0) or 1.0


def set_equal_3d_axes(ax, all_points):
    """Give a 3D axes equal, symmetric limits and label them in meters.

    Centers the chief at the origin of the plot box so trajectories from
    different runs stay visually comparable.

    Args:
        ax: The 3D axes to configure.
        all_points: Array of every plotted point, shape (N, 3).

    Returns:
        The half-width used for all three axis limits.
    """
    import numpy as np  # local import: keeps this module lightweight for path-only callers

    max_extent = np.max(np.abs(all_points))
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    return max_extent


def center_matplotlib_window(fig=None):
    """Center the current matplotlib figure window on the primary monitor.

    Call before plt.show().

    Best-effort only: some backends (e.g. headless "Agg") have no window
    to center. Failures are swallowed rather than raised, since window
    placement should never crash an eval run.

    Args:
        fig: The figure to center. Defaults to the current active figure.
    """
    try:
        import matplotlib.pyplot as plt

        fig = fig or plt.gcf()
        mgr = plt.get_current_fig_manager()
        window = getattr(mgr, "window", None)
        if window is None:
            return

        if hasattr(window, "winfo_screenwidth") and hasattr(window, "wm_geometry"):
            # TkAgg. Map the window first, then measure its real size;
            # the pre-map size under-predicts once the toolbar renders.
            window.update_idletasks()
            window.deiconify()
            window.update()
            width_px = window.winfo_width()
            height_px = window.winfo_height()
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()

            # Shrink slightly and bias upward so the bottom row clears the
            # taskbar, and constrain so the window never lands off-screen.
            shrink_w = 0.95
            shrink_h = 0.95 ** 2  # shorter than narrower; taskbar clearance
            shift_up_frac = 0.10
            new_w = int(width_px * shrink_w)
            new_h = int(height_px * shrink_h)
            x = max(0, (screen_w - new_w) // 2)
            y = max(0, (screen_h - new_h) // 2 - int(screen_h * shift_up_frac))
            y = max(0, y)
            x = min(x, max(0, screen_w - new_w))
            y = min(y, max(0, screen_h - new_h))
            window.wm_geometry(f"{new_w}x{new_h}+{x}+{y}")
            window.update()
            return

        # Non-Tk backends (Qt/WX): center using the figure's pixel size
        # and the primary monitor via GetSystemMetrics.
        width_px, height_px = fig.canvas.get_width_height()
        screen_w = screen_h = None
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
        except Exception:
            return
        x = max(0, (screen_w - width_px) // 2)
        y = max(0, (screen_h - height_px) // 2)
        x = min(x, max(0, screen_w - width_px))
        y = min(y, max(0, screen_h - height_px))
        try:  # Qt5Agg / QtAgg
            window.move(x, y)
            return
        except Exception:
            pass
        try:  # WXAgg
            window.SetPosition((x, y))
        except Exception:
            pass
    except Exception:
        pass


def plural_word(count, singular, plural=None):
    """Return the singular or plural form of a word for a given count.

    Args:
        count: The number of items.
        singular: Singular form, e.g. "episode".
        plural: Plural form. Defaults to singular + "s".

    Returns:
        singular if count == 1, else plural.
    """
    if plural is None:
        plural = singular + "s"
    return singular if count == 1 else plural


def format_duration(seconds):
    """Format a duration in seconds as "X min Y sec (Z.ZZ hrs)".

    Matches the wording every trainer already prints for "Total time"
    and "Stage complete", so run_metadata.json can record the same
    human-readable string next to the raw seconds instead of only the
    float, which nobody can read at a glance.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string like "34 min 7 sec (0.57 hrs)".
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    hours = seconds / 3600
    return f"{minutes} min {remaining_seconds} sec ({hours:.2f} hrs)"


def interaction_efficiency(epoch_history, threshold=0.8):
    """Cumulative training timesteps to first reach a dock-rate threshold.

    Matches SafeRL (2022)'s own Table 2 definition exactly: "the number
    of timesteps the agents had to interact with the environment before
    reaching an average success threshold of 80%", their 3D docking
    figure being 0.66e6. Deliberately independent of whatever this
    project's own CONSECUTIVE_PASSES_REQUIRED/pass threshold happens to
    be for a given run (usually stricter, and usually requiring more
    than one epoch in a row), so this is comparable to their number even
    when it differs from "this run's own pass point".

    Works on both a curriculum run's epoch_history (many stages, cumulative
    "timesteps" already spans the whole run) and a standalone run's (one
    stage), since both carry the same {"timesteps", "dock_rate", ...}
    schema per epoch.

    Args:
        epoch_history: A run's results.epoch_history list (from
            run_metadata.json or epoch_log.csv), each entry a dict with
            at least "timesteps" (cumulative across the whole run) and
            "dock_rate".
        threshold: Dock rate to reach, 0.8 by default to match SafeRL's
            own figure. Pass this project's own pass threshold instead
            for a same-bar-as-training comparison.

    Returns:
        The "timesteps" value of the first epoch whose dock_rate is >=
        threshold, or None if no epoch ever reached it (this run's best
        checkpoint may still be usable; it just never hit this
        particular bar within its budget).
    """
    for epoch in epoch_history:
        if epoch["dock_rate"] >= threshold:
            return epoch["timesteps"]
    return None


def interquartile_mean(values):
    """Interquartile mean (IQM): the mean of the middle 50% of the data.

    Trims the lowest and highest 25% and averages the rest. This is the
    robust statistic reported by the docking literature this project
    compares against (DRL for Space Controls (2024), arXiv 2405.12355),
    so reporting it too keeps citations on the same statistical footing.

    Degrades gracefully for small samples: under 4 values, nothing gets
    trimmed and this returns the plain mean.

    Args:
        values: A 1-D sequence of numbers (e.g. per-episode fuel).

    Returns:
        The IQM as a float, or nan for an empty input.
    """
    import numpy as np
    from scipy.stats import trim_mean

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(trim_mean(arr, 0.25))


def interquartile_mean_ci(values, confidence=95.0, num_resamples=10_000, seed=0):
    """Bootstrap confidence interval around the interquartile mean (IQM).

    Resamples the data many times to estimate the range where the true
    average (IQM) likely falls. Unlike standard deviation, this works well
    with win/loss data (bimodal distributions). Matches the method used in
    DRL for Space Controls (2024).

    Args:
        values: 1-D sequence of numbers (e.g., per-episode fuel costs).
        confidence: Interval width in percent (e.g., 95.0 for a 95% confidence interval).
        num_resamples: Bootstrap iterations. 10,000 is stable to ~0.001.
        seed: Fixed seed ensures same input always gives same interval.

    Returns:
        (iqm, low, high) tuple. Returns (iqm, iqm, iqm) for single values,
        or all-NaN for empty input (intervals don't make sense there).
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    iqm = interquartile_mean(arr)
    if arr.size == 1:
        return iqm, iqm, iqm

    rng = np.random.default_rng(seed)
    resample_iqms = [
        interquartile_mean(rng.choice(arr, size=arr.size, replace=True))
        for _ in range(num_resamples)
    ]
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(resample_iqms, [tail, 100.0 - tail])
    return iqm, float(low), float(high)


def paired_difference_ci(values_a, values_b, confidence=95.0,
                         num_resamples=10_000, seed=0):
    """Bootstrap confidence interval for the per-episode difference between two agents.

    Every eval script here pairs its episodes: both agents fly the exact
    same start states (see resolve_start_states), so episode i is the
    same task for each. This measures A minus B episode by episode,
    which cancels the shared "was this a hard start?" noise. Comparing
    each side's own confidence interval instead throws that pairing away, and two
    overlapping CIs do not mean the paired difference is zero.

    Args:
        values_a: Per-episode values for A (e.g. fuel), episode order.
        values_b: Per-episode values for B, same episodes in the same
            order. Must be the same length as values_a.
        confidence: Interval width in percent, 95.0 for a 95% confidence interval.
        num_resamples: Bootstrap resamples over the difference vector.
        seed: Fixed so the same data always yields the same interval.

    Returns:
        A (mean_difference, low, high) tuple. The difference is A minus
        B, so a negative interval that excludes zero means A spent less.
        All-nan for empty input.

    Raises:
        ValueError: If the two sequences differ in length, which means
            they are not paired.
    """
    import numpy as np

    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    if arr_a.size != arr_b.size:
        raise ValueError(
            f"Paired comparison needs equal-length inputs, got "
            f"{arr_a.size} and {arr_b.size}. These results are not paired."
        )
    if arr_a.size == 0:
        return float("nan"), float("nan"), float("nan")

    differences = arr_a - arr_b
    mean_difference = float(np.mean(differences))
    if arr_a.size == 1:
        return mean_difference, mean_difference, mean_difference

    rng = np.random.default_rng(seed)
    resample_means = [
        float(np.mean(rng.choice(differences, size=differences.size, replace=True)))
        for _ in range(num_resamples)
    ]
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(resample_means, [tail, 100.0 - tail])
    return mean_difference, float(low), float(high)


def mcnemar_exact(wins_a, wins_b):
    """Exact McNemar test on paired win/loss outcomes.

    The paired counterpart to comparing two win rates. Only the
    DISCORDANT episodes carry information about a difference: the ones
    where exactly one agent won. Episodes both agents won (or both lost)
    say nothing about which is better, so they drop out. Comparing two
    per-side Wilson intervals ignores this structure entirely.

    Uses the exact binomial test rather than the chi-squared
    approximation, since the discordant count here is usually tiny
    (often under 5), where the approximation is unreliable.

    Args:
        wins_a: Per-episode booleans for A, episode order.
        wins_b: Per-episode booleans for B, same episodes in the same order.

    Returns:
        An (a_only, b_only, p_value) tuple. a_only is how many episodes
        A won that B lost, b_only the reverse. p_value is two-sided; it
        is 1.0 when there are no discordant episodes, since identical
        outcomes are no evidence of a difference.

    Raises:
        ValueError: If the two sequences differ in length, which means
            they are not paired.
    """
    if len(wins_a) != len(wins_b):
        raise ValueError(
            f"Paired comparison needs equal-length inputs, got "
            f"{len(wins_a)} and {len(wins_b)}. These results are not paired."
        )

    a_only = sum(1 for a, b in zip(wins_a, wins_b) if a and not b)
    b_only = sum(1 for a, b in zip(wins_a, wins_b) if b and not a)
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0

    # Two-sided exact binomial test at p=0.5: probability of a split at
    # least this lopsided, counted from both tails.
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return a_only, b_only, p_value


def build_discrete_thrust_colormap(max_thrust, bin_step=0.25):
    """Build a discrete colormap and norm for thrust-magnitude coloring.

    Bin edges round up to whole multiples of bin_step, so the colorbar
    always shows clean ticks (0, 0.25, 0.5, ...) regardless of
    max_thrust. Bin count grows automatically to cover it.

    First bin ("coasting") is neutral grey; every bin above is an
    increasingly dark blue from Matplotlib's "Blues".

    Args:
        max_thrust: The highest thrust magnitude the colormap needs to
            cover, in N.
        bin_step: Width of each discrete bin, in N.

    Returns:
        A (cmap, norm) tuple: a matplotlib ListedColormap and the
        matching BoundaryNorm.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex

    num_bins = max(1, int(np.ceil(max_thrust / bin_step)))
    bin_edges = np.arange(num_bins + 1) * bin_step

    blues = plt.get_cmap("Blues")
    if num_bins == 1:
        bin_colors = ["#969696"]
    else:
        # Sample 40%-98% of the Blues range; below 40% was too pale to see.
        bin_colors = ["#969696"] + [
            to_hex(blues(0.40 + 0.58 * i / max(1, num_bins - 2)))
            for i in range(num_bins - 1)
        ]

    cmap = ListedColormap(bin_colors)
    norm = BoundaryNorm(bin_edges, cmap.N)
    return cmap, norm


# Backend names with no GUI window to show. Set via MPLBACKEND or
# matplotlib.use(); makes a run headless, skipping plt.show().
_HEADLESS_BACKENDS = frozenset((
    "agg", "pdf", "ps", "svg", "svgz", "cairo", "pgf", "template",
))


def should_show_plot(show=True):
    """Decide whether a figure's window should be shown.

    Combines a caller's show flag (e.g. a script's SHOW_PLOTS
    constant) with whether the active Matplotlib backend has any window
    to show at all. False either way means no window: show=False always
    wins, and a headless backend (see _HEADLESS_BACKENDS, e.g. one set
    via MPLBACKEND=Agg) always wins regardless of show. Saving to disk
    is never affected by this, only whether a window pops up.

    Args:
        show: The caller's preference. Defaults to True so existing
            callers that never pass this keep behaving exactly as before.

    Returns:
        True only if show is True and the backend has a window.
    """
    import matplotlib
    return bool(show) and matplotlib.get_backend().lower() not in _HEADLESS_BACKENDS


def save_and_show_figure(base_dir, subfolder, filename_prefix, description="plot",
                         show=True):
    """Save the current figure under a timestamped name, then show it.

    Always saves regardless of show; show only controls whether a
    window pops up.

    Two independent ways to suppress the window, either is enough on its
    own: pass show=False (every eval script exposes this as its own
    SHOW_PLOTS constant), or run under a headless backend (see
    _HEADLESS_BACKENDS), e.g.:
        MPLBACKEND=Agg python drift_vs_drift_test.py          (bash)
        $env:MPLBACKEND="Agg"; python drift_vs_drift_test.py  (PowerShell)
    The backend check avoids a "cannot show the figure" warning per plot
    when show=True but no display exists (SSH or automated build). SHOW_PLOTS is for the
    opposite case: a normal display where the user just wants no popup.

    Args:
        base_dir: The script's BASE_DIR (a pathlib.Path). Figures go
            under base_dir / "saved_figures" / subfolder.
        subfolder: Per-script folder name, e.g. "drift_vs_drift_test".
        filename_prefix: Start of the filename; timestamp and ".png"
            are appended automatically.
        description: Noun for the printed confirmation line, e.g.
            "approach direction plot".
        show: False to skip popping up a window for this figure. The
            PNG is still written either way.

    Returns:
        The pathlib.Path the figure was written to.
    """
    import matplotlib.pyplot as plt

    save_dir = base_dir / "saved_figures" / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{filename_prefix}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {description}:\n{save_path}")
    if should_show_plot(show):
        center_matplotlib_window()
        plt.show()
    plt.close()
    return save_path


def fuel_statistics(fuels, fuels_l1):
    """Compute the standard fuel summary for one evaluated agent.

    Reports mean, median, IQM (with a bootstrap 95% confidence interval), min, max, and
    25th/75th percentile, for both L1 (primary) and L2 Delta-v.

    Args:
        fuels: Per-episode L2-norm Delta-v, what a single gimbaled
            thruster would have spent. Secondary metric.
        fuels_l1: Per-episode L1-norm Delta-v, same ordering.
            Proportional to propellant used by the three
            fixed per-axis thrusters, so this is the primary metric.

    Returns:
        A dict ready to splat into a result dict, using the key names
        the comparison table expects ("mean_fuel", "iqm_fuel_l1", etc).
        The confidence interval bounds are "iqm_fuel_lo"/"iqm_fuel_hi"
        and their _l1 twins.
    """
    import numpy as np

    stats = {}
    for values, suffix in ((fuels, ""), (fuels_l1, "_l1")):
        stats[f"mean_fuel{suffix}"] = float(np.mean(values))
        stats[f"median_fuel{suffix}"] = float(np.median(values))
        iqm, iqm_lo, iqm_hi = interquartile_mean_ci(values)
        stats[f"iqm_fuel{suffix}"] = iqm
        stats[f"iqm_fuel{suffix}_lo"] = iqm_lo
        stats[f"iqm_fuel{suffix}_hi"] = iqm_hi
        stats[f"min_fuel{suffix}"] = float(np.min(values))
        stats[f"max_fuel{suffix}"] = float(np.max(values))
        stats[f"p25_fuel{suffix}"] = float(np.percentile(values, 25))
        stats[f"p75_fuel{suffix}"] = float(np.percentile(values, 75))
    return stats


def plot_fuel_comparison(results, colors, base_dir, subfolder, filename_prefix, seed=None,
                         show=True):
    """Box plot of per-episode fuel usage for one or more evaluated agents.

    Works with a single result (one box) or a paired A/B comparison (two
    boxes side by side); any script with only one checkpoint to show, or
    the usual two-checkpoint comparison, can call this the same way.

    Args:
        results: List of 1+ result dicts from an evaluate_*() function,
            each must carry "label" and "fuels" (raw per-episode L2 Delta-v).
        colors: Dict mapping tag ("A", "B", ...) to a color, same as the
            trajectory/diagnostics plots use.
        base_dir: The script's BASE_DIR, passed through to save_and_show_figure.
        subfolder: Per-script output folder name.
        filename_prefix: Start of the saved filename.
        seed: Optional seed to note in the title for reproducibility.
        show: Passed through to save_and_show_figure; False saves the
            PNG without popping up a window.

    Returns:
        The pathlib.Path the figure was written to.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tags = list(colors.keys())[:len(results)]
    positions = list(range(1, len(results) + 1))
    # Plot L1 (propellant the fixed thrusters use). Falls back
    # to L2 for older result dicts that predate fuels_l1.
    data = [r.get("fuels_l1") or r["fuels"] for r in results]
    labels = [r["label"] for r in results]

    fig, ax = plt.subplots(figsize=(4 + 2.5 * len(results), 6))
    box = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                      showmeans=True,
                      meanprops={"marker": "D", "markerfacecolor": "white",
                                 "markeredgecolor": "black", "markersize": 7})
    for patch, tag in zip(box["boxes"], tags):
        patch.set_facecolor(colors[tag])
        patch.set_alpha(0.6)

    # Overlay raw per-episode points so a small sample doesn't hide
    # behind a box that looks more confident than the data is.
    rng = np.random.default_rng(0)
    for pos, values in zip(positions, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values,
                   color="black", alpha=0.4, s=15, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta-v used, per-axis sum (L1, m/s)")
    title = "Delta-v usage per episode"
    if seed is not None:
        title += f" (seed {seed})"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return save_and_show_figure(base_dir, subfolder, filename_prefix, "fuel comparison plot",
                               show=show)


def plot_paired_trajectories_3d(
    result_a,
    result_b,
    colors,
    base_dir,
    subfolder,
    filename_prefix,
    title,
    win_outcome_value="win",
    color_by_thrust=True,
    thrust_colormap="discrete",
    thrust_burst_markers=True,
    bright_thrust_highlights=False,
    bright_thrust_color="#FF2A00",
    thrust_burst_fraction=0.6,
    colorbar_at_bottom=False,
    seed=None,
    show=True,
):
    """Save a 3D trajectory plot with both agents' episodes overlaid.

    Each agent gets its own color; wins are drawn as solid lines, losses
    as dashed, so both which-agent and win/loss are visible at once.
    NOT used by drift_vs_nodrift_test.py: its plot_comparison() is a
    deliberately separate, local implementation (per-episode gradient
    coloring, agent-specific markers and linestyle, a start-marker
    jitter, and a dash-pattern technique for the thrust-colored mode, none
    of which the other two callers need). If you fix a rendering bug or
    add a feature here, check whether drift_vs_nodrift_test.py's local
    copy needs the same change.
    Shared by every paired-comparison script (drift-vs-drift,
    drift-vs-nodrift, nodrift-vs-nodrift); the small differences between
    them (what counts as a "win", the plot title, and a handful of
    thrust-drawing style settings) are the arguments below.

    Args:
        result_a: Result dict for agent A. Must carry "trajectories",
            "thrusts", "outcomes" (one entry per recorded episode),
            "label", "checkpoint", "wins", and "total".
        result_b: Same, for agent B.
        colors: Dict mapping tag ("A", "B") to a plot color.
        base_dir: The calling script's BASE_DIR, passed through to
            save_and_show_figure.
        subfolder: Per-script output folder name for the saved figure.
        filename_prefix: Start of the saved figure's filename.
        title: Plot title, without the win/loss counts line (those are
            appended automatically from result_a/result_b).
        win_outcome_value: The outcome string that means "this episode
            docked successfully". Different scripts use different words
            ("win" for drift episodes, "direct_dock" for nodrift ones).
        color_by_thrust: True colors each trajectory segment by thrust
            magnitude instead of a flat per-agent color, and enables the
            burst-marker and colorbar options below.
        thrust_colormap: Colormap name passed to build_thrust_colormap()
            when color_by_thrust is True.
        thrust_burst_markers: True stamps a triangle marker on any step
            thrusting above thrust_burst_fraction of the episode's max,
            so a brief hard thrust stands out instead of blending into the
            line.
        bright_thrust_highlights: True colors burst markers a single
            bright, enlarged color for maximum visibility; False colors
            them by thrust magnitude instead (subtler).
        bright_thrust_color: Hex color used when bright_thrust_highlights
            is True.
        thrust_burst_fraction: Fraction of an episode's max thrust above
            which a step counts as a "burst" and gets a marker.
        colorbar_at_bottom: True draws the thrust colorbar as a
            horizontal bar below the axes instead of a vertical bar at
            the right (Matplotlib's default).
        seed: Optional resolved seed, noted in the title so the figure
            records which run produced it.
        show: Passed through to save_and_show_figure; False saves the
            PNG without popping up a window.

    Returns:
        The pathlib.Path the figure was written to.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    labels_used = set()
    burst_labeled = False
    thrust_line = None  # last Line3DCollection drawn, used to anchor the colorbar

    if color_by_thrust:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        # Build one color scale from the hardest thrust across BOTH
        # agents, so the same shade means the same thrust on every line.
        max_thrust = max_thrust_across_results(result_a, result_b)
        thrust_cmap, thrust_norm = build_thrust_colormap(thrust_colormap, max_thrust)

    for tag, result in (("A", result_a), ("B", result_b)):
        for traj, thrust, outcome in zip(result["trajectories"], result["thrusts"], result["outcomes"]):
            traj = np.array(traj)
            if len(traj) < 2:
                continue
            is_win = outcome == win_outcome_value
            key = (tag, is_win)
            label = None
            if key not in labels_used:
                label = f"{result['label']} ({'win' if is_win else 'loss'})"
                labels_used.add(key)

            if color_by_thrust:
                # Agent identity is still shown via the start dot's
                # color; thrust intensity comes from segment color.
                points = traj.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_thrust = np.array(thrust[1:len(traj)])
                linestyle = "-" if is_win else (0, (2, 2))
                lc = Line3DCollection(segments, cmap=thrust_cmap, norm=thrust_norm,
                                       alpha=0.9, linestyle=linestyle)
                lc.set_array(segment_thrust)
                if thrust_burst_markers:
                    # Thicker line for harder thrust, not just a color
                    # change, so a brief hard thrust stands out at a glance.
                    lc.set_linewidth(1.5 + 4.5 * (segment_thrust / max_thrust))
                else:
                    lc.set_linewidth(2.0)
                ax.add_collection3d(lc)
                thrust_line = lc

                if thrust_burst_markers:
                    burst_mask = segment_thrust >= thrust_burst_fraction * max_thrust
                    if np.any(burst_mask):
                        burst_points = traj[1:][burst_mask]
                        burst_values = segment_thrust[burst_mask]
                        burst_label = None
                        if not burst_labeled:
                            burst_label = "Hard burn"
                            burst_labeled = True
                        if bright_thrust_highlights:
                            ax.scatter(burst_points[:, 0], burst_points[:, 1], burst_points[:, 2],
                                       color=bright_thrust_color, marker="^", s=90,
                                       edgecolors="black", linewidths=0.6,
                                       alpha=1.0, zorder=6, label=burst_label)
                        else:
                            ax.scatter(burst_points[:, 0], burst_points[:, 1], burst_points[:, 2],
                                       c=burst_values, cmap=thrust_cmap, norm=thrust_norm,
                                       marker="^", s=55, edgecolors="black", linewidths=0.5,
                                       alpha=0.95, zorder=6, label=burst_label)

                ax.scatter(*traj[0], color=colors[tag], s=25, alpha=0.9,
                           edgecolors="black", linewidths=0.5, label=label)
            else:
                linestyle = "-" if is_win else "--"
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[tag],
                         linestyle=linestyle, linewidth=1.2, alpha=0.6, label=label)
                ax.scatter(*traj[0], color=colors[tag], s=20, alpha=0.9)

    if thrust_line is not None:
        if colorbar_at_bottom:
            # Smaller shrink and larger pad than the vertical bar, so it
            # doesn't overlap the 3D axes' tick labels.
            cbar = fig.colorbar(thrust_line, ax=ax, location="bottom", orientation="horizontal",
                                shrink=0.4, pad=0.05)
        else:
            cbar = fig.colorbar(thrust_line, ax=ax, location="right", orientation="vertical",
                                shrink=0.6, pad=0.04)
        cbar.set_label("Thrust magnitude (Newtons, L2 norm per step)", labelpad=10)

    ax.scatter(0, 0, 0, marker="x", color="black", s=150, linewidths=2.5,
               zorder=5, label="Chief (target)")

    all_points = np.concatenate([
        np.array(t) for r in (result_a, result_b) for t in r["trajectories"] if len(t) > 0
    ])
    set_equal_3d_axes(ax, all_points)
    seed_note = f" (seed {seed})" if seed is not None else ""
    ax.set_title(
        f"{title}{seed_note}\n"
        f"A: {result_a['checkpoint']}  ({result_a['wins']}/{result_a['total']})\n"
        f"B: {result_b['checkpoint']}  ({result_b['wins']}/{result_b['total']})",
        fontsize=10,
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    # tight_layout() alone can still let a tall 3D box's title clip off
    # the top of the window; reserve explicit headroom for it.
    fig.subplots_adjust(top=0.92)

    return save_and_show_figure(base_dir, subfolder, filename_prefix, "trajectory plot",
                               show=show)


def plot_paired_diagnostics(
    result_a,
    result_b,
    colors,
    base_dir,
    subfolder,
    filename_prefix,
    coast_threshold,
    diag_num_episodes=5,
    seed=None,
    show=True,
):
    """Save a 2x2 per-step diagnostics figure with both agents overlaid.

    NOT used by drift_vs_nodrift_test.py: its plot_diagnostics() is a
    deliberately separate, local implementation (per-episode win/loss
    gradient coloring plus agent-specific linestyle). If you fix a
    rendering bug or add a feature here, check whether
    drift_vs_nodrift_test.py's local copy needs the same change.

    Panels: thrust vs. time, thrust vs. distance to chief, distance vs.
    time, and speed vs. time, for the first diag_num_episodes recorded
    episodes per agent. Where the 3D trajectory plot shows *where* each
    agent flies, this figure shows *how*: a fuel-efficient agent reads as
    long near-zero-thrust stretches with short braking thrusts, while a
    fuel-hungry one thrusts at nearly every distance.

    Args:
        result_a: Result dict for agent A. Must carry "trajectories",
            "thrusts", "speeds" (one entry per recorded episode) and
            "label".
        result_b: Same, for agent B.
        colors: Dict mapping tag ("A", "B") to a plot color, matching the
            3D trajectory plot's colors.
        base_dir: The calling script's BASE_DIR, passed through to
            save_and_show_figure.
        subfolder: Per-script output folder name for the saved figure.
        filename_prefix: Start of the saved figure's filename.
        coast_threshold: Thrust magnitude (Newtons) below which a step
            counts as coasting; drawn as a reference line on both thrust
            panels.
        diag_num_episodes: How many of each agent's first recorded
            episodes to overlay.
        seed: Optional resolved seed, noted in the title so the figure
            records which run produced it.
        show: Passed through to save_and_show_figure; False saves the
            PNG without popping up a window.

    Returns:
        The pathlib.Path the figure was written to.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_thrust_time, ax_thrust_dist), (ax_dist_time, ax_speed_time) = axes

    labels_used = set()

    for tag, result in (("A", result_a), ("B", result_b)):
        n_episodes = min(diag_num_episodes, len(result["trajectories"]))
        for i in range(n_episodes):
            traj = np.array(result["trajectories"][i])
            thrust = np.array(result["thrusts"][i])
            speed = np.array(result["speeds"][i])
            if len(traj) < 2:
                continue
            distance = np.linalg.norm(traj, axis=1)
            time = np.arange(len(traj))

            label = None
            if tag not in labels_used:
                label = result["label"]
                labels_used.add(tag)

            ax_thrust_time.plot(time, thrust, color=colors[tag], alpha=0.5,
                                linewidth=1.0, label=label)
            ax_thrust_dist.plot(distance, thrust, color=colors[tag], alpha=0.5,
                                linewidth=1.0)
            ax_dist_time.plot(time, distance, color=colors[tag], alpha=0.5,
                              linewidth=1.0)
            ax_speed_time.plot(time, speed, color=colors[tag], alpha=0.5,
                               linewidth=1.0)

    # Mark the coast threshold on both thrust panels, matching the line
    # used in any printed coast fraction.
    for ax in (ax_thrust_time, ax_thrust_dist):
        ax.axhline(coast_threshold, color="black", linestyle=":",
                   linewidth=1.0, alpha=0.7)

    ax_thrust_time.set_xlabel("Time (steps = seconds)")
    ax_thrust_time.set_ylabel("Thrust magnitude (Newtons, L2 norm)")
    ax_thrust_time.set_title("Thrust vs. time")

    ax_thrust_dist.set_xlabel("Distance to chief (m)")
    ax_thrust_dist.set_ylabel("Thrust magnitude (Newtons, L2 norm)")
    ax_thrust_dist.set_title("Thrust vs. distance")
    ax_thrust_dist.invert_xaxis()  # approach reads left to right: far -> docked

    ax_dist_time.set_xlabel("Time (steps = seconds)")
    ax_dist_time.set_ylabel("Distance to chief (m)")
    ax_dist_time.set_title("Distance vs. time")

    ax_speed_time.set_xlabel("Time (steps = seconds)")
    ax_speed_time.set_ylabel("Speed (m/s)")
    ax_speed_time.set_title("Speed vs. time")

    ax_thrust_time.legend(loc="upper right")
    seed_note = f" (seed {seed})" if seed is not None else ""
    fig.suptitle(
        f"Per-step diagnostics: {result_a['label']} vs {result_b['label']}{seed_note}\n"
        f"First {diag_num_episodes} {plural_word(diag_num_episodes, 'episode')} each\n"
        f"Dotted line = coast threshold {coast_threshold} Newtons",
        fontsize=11,
    )
    plt.tight_layout()

    return save_and_show_figure(base_dir, subfolder, filename_prefix, "diagnostics plot",
                               show=show)


def plot_paired_approach_directions(result_a, result_b, colors, base_dir,
                                    subfolder, filename_prefix,
                                    check_distance, seed=None, show=True):
    """Scatter each episode's approach direction for two evaluated agents.

    Each point is one episode's approach direction (unit vector from
    chief to deputy), at the first step within check_distance, plotted
    as azimuth vs. elevation. A tight cluster means the agent funnels
    every approach into one memorized corridor; spread means each
    episode keeps its own starting direction.

    Args:
        result_a: Result dict for side A. Needs "entry_directions",
            "approach_spread_deg", and "label".
        result_b: Same, for side B.
        colors: Dict mapping "A" and "B" to matplotlib colors.
        base_dir: The calling script's BASE_DIR (a pathlib.Path).
        subfolder: Figure subfolder name, see save_and_show_figure.
        filename_prefix: Figure filename prefix, see save_and_show_figure.
        check_distance: Distance from the chief, in meters, at which each
            approach direction was sampled. Shown in the title.
        seed: Optional resolved seed, added to the title when given. Pass
            None to leave it out.
        show: Passed through to save_and_show_figure; False saves the
            PNG without popping up a window.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    for tag, result in (("A", result_a), ("B", result_b)):
        directions = np.array(result["entry_directions"])
        if len(directions) == 0:
            continue
        azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
        elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
        ax.scatter(azimuth, elevation, color=colors[tag], alpha=0.75, s=40,
                   edgecolors="black", linewidths=0.5,
                   label=f"{result['label']} "
                         f"(spread {result['approach_spread_deg']:.1f} deg)")

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    seed_note = f" (seed {seed})" if seed is not None else ""
    ax.set_title(
        f"Approach direction at {check_distance:.0f} m from chief{seed_note}\n"
        f"{result_a['label']} vs {result_b['label']} "
        "(tight cluster = single corridor, spread = direction preserved)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_show_figure(base_dir, subfolder, filename_prefix,
                         "approach direction plot", show=show)


# Rows of the paired comparison table, in print order. Each entry is
# (row label, function rendering one side's cell from its result dict).
# A row is skipped when a result dict lacks the key it needs.
_COMPARISON_TABLE_ROWS = [
    ("Checkpoint", lambda r: str(r["checkpoint"])),
    # Includes the 95% Wilson interval alongside the raw rate.
    ("Win rate", lambda r: format_win_rate(r["wins"], r["total"])),
    # L1 is the real fuel cost (three fixed thrusters). L2 is the
    # gimbaled-equivalent Delta-v.
    ("Delta-v, per-axis sum (L1)",
     lambda r: f"mean {r['mean_fuel_l1']:.3f}  median {r['median_fuel_l1']:.3f}"),
    ("Delta-v, vector norm (L2)",
     lambda r: f"mean {r['mean_fuel']:.3f}  median {r['median_fuel']:.3f}"),
    ("Mean episode length", lambda r: f"{r['mean_timesteps']:.1f} steps"),
    # Report simulated seconds. Drift coast steps can cover more time than
    # one step, while nodrift uses one second per step.
    ("Time to dock",
     lambda r: f"mean {r['mean_time_to_dock']:.0f}s  median {r['median_time_to_dock']:.0f}s"),
    # Total episode reward under the reward function.
    ("Mean episode reward",
     lambda r: f"mean {r['mean_episode_reward']:.3f}  "
               f"median {r['median_episode_reward']:.3f}"),
    # Divides each drift episode by its chain length so it's comparable
    # to a nodrift episode's single reward. Rough, not exact.
    ("Mean episode reward (weighted)",
     lambda r: f"mean {r['mean_episode_reward_weighted']:.3f}  "
               f"median {r['median_episode_reward_weighted']:.3f}"),
    # Share of steps below the coast thrust threshold.
    ("Coast fraction",
     lambda r: f"{100 * r['coast_fraction']:.1f}% of steps"),
    # Share of steps below the tighter true-coast threshold. Absent from
    # nodrift results, so that row drops out there.
    ("True coast fraction",
     lambda r: f"{100 * r['true_coast_fraction']:.1f}% of steps"),
    # Share of steps with any action axis near max thrust.
    ("Saturation fraction",
     lambda r: f"{100 * r['saturation_fraction']:.1f}% of steps"),
    # Matches the metrics reported by DRL for Space Controls (2024,
    # arXiv 2405.12355), Table II.
    ("Speed limit violation",
     lambda r: f"{100 * r['violation_fraction']:.1f}% of steps"),
    # Unweighted by episode length: mean of each episode's own violation
    # rate, versus the step-weighted row above.
    ("Speed limit violation (per-episode mean)",
     lambda r: f"{100 * r['violation_fraction_episode_mean']:.1f}% of episode, avg"),
    ("Mean final speed",
     lambda r: f"{r['mean_final_speed']:.4f} m/s"),
    # How often the one-step safety check kept the model action vs.
    # overrode it. Only present when USE_SAFE_ACTION is on.
    ("Safety check: kept model action",
     lambda r: (f"{100 * r['safe_action_model_fraction']:.1f}% of "
                f"{r['safe_action_checked_steps']} checked steps")),
    ("Safety check: overrode action",
     lambda r: (f"{100 * (r['safe_action_zero_thrust_fraction'] + r['safe_action_braking_fraction']):.1f}%"
                f"  (zero {100 * r['safe_action_zero_thrust_fraction']:.1f}%, "
                f"brake {100 * r['safe_action_braking_fraction']:.1f}%)")),
    # Mean pairwise angle between episodes' approach unit vectors.
    ("Approach direction spread",
     lambda r: f"{r['approach_spread_deg']:.1f} deg"),
    ("Top failure", lambda r: format_top_failure(r["failure_counts"])),
]


def print_paired_tests(result_a, result_b, win_outcome_value="win"):
    """Print the paired A-vs-B tests that use the shared start states.

    The table above reports each side on its own. These two lines test
    the difference directly, which is what the paired design supports:
    both agents flew identical start states, so episode i is the same
    task for each. Reading the table's two separate intervals for
    overlap is not a valid test, and it ignores the pairing that makes
    this comparison sensitive in the first place.

    Skipped with no output when the results lack the per-episode arrays
    (older result dicts) or have mismatched episode counts, since
    neither test is meaningful then.

    Args:
        result_a: Result dict for side A. Needs "fuels_l1" and
            "all_outcomes", both one entry per episode.
        result_b: Same, for side B.
        win_outcome_value: The "all_outcomes" string meaning a win.
    """
    fuels_a = result_a.get("fuels_l1")
    fuels_b = result_b.get("fuels_l1")
    outcomes_a = result_a.get("all_outcomes")
    outcomes_b = result_b.get("all_outcomes")
    if not fuels_a or not fuels_b or not outcomes_a or not outcomes_b:
        return
    if len(fuels_a) != len(fuels_b) or len(outcomes_a) != len(outcomes_b):
        return

    label_a, label_b = result_a["label"], result_b["label"]
    print(f"\nPaired tests ({label_a} minus {label_b}, same start states):")

    mean_difference, low, high = paired_difference_ci(fuels_a, fuels_b)
    # An interval clear of zero means the gap survives the episode-level
    # noise; one straddling zero means this sample cannot tell them apart.
    verdict = "excludes 0" if not (low <= 0.0 <= high) else "includes 0, not resolved"
    print(f"  Delta-v (L1) difference:  mean {mean_difference:+.3f} m/s  "
          f"95% confidence interval [{low:+.3f}, {high:+.3f}]  ({verdict})")

    wins_a = [outcome == win_outcome_value for outcome in outcomes_a]
    wins_b = [outcome == win_outcome_value for outcome in outcomes_b]
    a_only, b_only, p_value = mcnemar_exact(wins_a, wins_b)
    if a_only + b_only == 0:
        print(f"  Win rate difference:      no episode split the two agents, "
              f"nothing to test")
    else:
        print(f"  Win rate difference:      {label_a} won {a_only} that "
              f"{label_b} lost, {label_b} won {b_only} that {label_a} lost  "
              f"(McNemar exact p = {p_value:.3f})")


def print_paired_comparison_table(result_a, result_b, seed, num_episodes,
                                  win_outcome_value="win"):
    """Print the side-by-side metric table for two evaluated agents.

    Followed by the paired A-vs-B tests (see print_paired_tests), each
    side's full failure-reason breakdown (not just the table's single
    "Top failure" row), and each side's fuel percentile breakdown.

    Rows come from _COMPARISON_TABLE_ROWS above; a row needing a metric
    a result doesn't carry is skipped. Column widths are measured from
    the actual content, so a long checkpoint name never runs columns together.

    Args:
        result_a: Result dict for side A. Needs "label", "checkpoint", and
            whichever metrics the rows above read.
        result_b: Same, for side B.
        seed: Resolved seed the paired episodes ran with, for the header.
        num_episodes: Episodes evaluated per side, for the header.
        win_outcome_value: The "all_outcomes" string meaning a win, used
            by the paired win-rate test. "win" for drift-style results,
            "direct_dock" for nodrift_full_test.py's.
    """
    rendered = []
    for label, render in _COMPARISON_TABLE_ROWS:
        try:
            rendered.append((label, render(result_a), render(result_b)))
        except KeyError:
            # This pair of results does not track this metric; skip the row.
            continue

    label_a, label_b = result_a["label"], result_b["label"]
    name_width = max(len("Metric"), *(len(r[0]) for r in rendered))
    a_width = max(len(label_a), *(len(r[1]) for r in rendered))
    b_width = max(len(label_b), *(len(r[2]) for r in rendered))

    print(f"\nComparison ({num_episodes} paired "
          f"{plural_word(num_episodes, 'episode')}, seed {seed}):")
    print(f"  {'Metric':<{name_width}}  {label_a:<{a_width}}  {label_b:<{b_width}}")
    print(f"  {'-' * (name_width + a_width + b_width + 4)}")
    for name, a_val, b_val in rendered:
        print(f"  {name:<{name_width}}  {a_val:<{a_width}}  {b_val:<{b_width}}")

    print_paired_tests(result_a, result_b, win_outcome_value=win_outcome_value)

    # Full breakdown per side, since a checkpoint can fail more than one
    # way and the top-failure summary alone would hide that.
    if "failure_counts" in result_a and "failure_counts" in result_b:
        for result in (result_a, result_b):
            print_failure_breakdown(
                f"{result['label']} ({result['checkpoint']})",
                result["failure_counts"],
            )

    # Percentiles print per side rather than in the table above, since the
    # full min/25th/median/75th/max/IQM string is far too wide for a column.
    for result in (result_a, result_b):
        print(f"\nFuel percentiles (Delta-v, m/s), "
              f"{result['label']} ({result['checkpoint']}):")
        # L1 (what the fixed thrusters spend) first, L2 (the
        # gimbaled-equivalent Delta-v) second, matching the table above.
        for norm_name, suffix in (("per-axis sum (L1)", "_l1"),
                                  ("vector norm  (L2)", "")):
            # The bracketed range after iqm is a bootstrap 95% confidence interval, absent
            # on results predating interquartile_mean_ci.
            iqm_ci = ""
            if f"iqm_fuel{suffix}_lo" in result:
                iqm_ci = (f" [95% confidence interval {result[f'iqm_fuel{suffix}_lo']:.3f}, "
                          f"{result[f'iqm_fuel{suffix}_hi']:.3f}]")
            # Width 15 aligns the two labels' columns with each other.
            print(f"  {norm_name + ':':<15} "
                  f"min {result[f'min_fuel{suffix}']:.3f}  "
                  f"25th {result[f'p25_fuel{suffix}']:.3f}  "
                  f"median {result[f'median_fuel{suffix}']:.3f}  "
                  f"75th {result[f'p75_fuel{suffix}']:.3f}  "
                  f"max {result[f'max_fuel{suffix}']:.3f}  "
                  f"iqm {result[f'iqm_fuel{suffix}']:.3f}{iqm_ci}")


def build_export_output_name(checkpoint_path, model, override_name, name_prefix="docking_policy"):
    """Build a descriptive .onnx filename from the checkpoint itself.

    Args:
        checkpoint_path: Full path to the .zip checkpoint that was loaded.
        model: The loaded PPO model, used to read the observation dimension.
        override_name: If given, used as-is (a ".onnx" suffix is added if
            missing) instead of the auto-generated name.
        name_prefix: Prefix identifying the export kind, e.g.
            "docking_policy" for a full nodrift export, or
            "docking_policy_drift_network" for a drift network-only
            export, so the two can never be mistaken for each other.

    Returns:
        A filename (no directory) ending in ".onnx".
    """
    if override_name:
        return override_name if override_name.endswith(".onnx") else override_name + ".onnx"

    run_kind = "curriculum" if describe_run_type(checkpoint_path) == "curriculum" else "standalone"
    run_number = extract_run(checkpoint_path)
    stage_number = extract_stage(checkpoint_path)
    obs_dim = model.observation_space.shape[0]

    run_part = f"{run_number}" if run_number != -1 else "unknownrun"
    stage_part = "final" if run_kind == "standalone" else f"stage{stage_number}"

    return f"{name_prefix}_{run_kind}_{run_part}_{stage_part}_obs{obs_dim}.onnx"


def build_onnxable_policy(policy):
    """Wrap an SB3 policy into a plain module for torch.onnx.export.

    The wrapper takes one observation tensor and returns one action
    tensor, discarding the value estimate and log-probability that
    policy.forward() also returns during training.

    deterministic=True makes the continuous action space (a Box, see
    docking_env.py's action_space) return the distribution's mean instead
    of a stochastic sample, matching how every eval script in this repo
    calls model.predict(deterministic=True).

    Args:
        policy: The .policy attribute of a loaded SB3 PPO model.

    Returns:
        A torch.nn.Module ready to pass to torch.onnx.export.
    """
    import torch  # local import: keeps this module lightweight for path-only callers

    class OnnxablePolicy(torch.nn.Module):
        """Wraps an SB3 policy so forward() returns only the action."""

        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation):
            """Return the deterministic action for one observation."""
            actions, _values, _log_prob = self.policy(observation, deterministic=True)
            return actions

    onnxable_policy = OnnxablePolicy(policy)
    onnxable_policy.eval()
    return onnxable_policy


def verify_onnx_export(onnx_path, onnxable_policy, dummy_input):
    """Check an exported ONNX file against the original PyTorch module.

    Runs the same input through both the original PyTorch module and the
    exported ONNX graph, and confirms they agree. Catches export bugs
    immediately instead of only discovering a broken graph after deployment.

    Args:
        onnx_path: Path to the just-written .onnx file.
        onnxable_policy: The wrapped policy module used for the export.
        dummy_input: The same input tensor used to trace the export.

    Raises:
        RuntimeError: If the PyTorch and ONNX outputs do not match.
    """
    import numpy as np  # local import: keeps this module lightweight for path-only callers
    import torch
    import onnxruntime as ort

    with torch.no_grad():
        torch_output = onnxable_policy(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

    if not np.allclose(torch_output, onnx_output, atol=1e-5):
        raise RuntimeError(
            f"ONNX export verification failed: PyTorch output {torch_output} "
            f"does not match ONNX output {onnx_output}. Do not use this file."
        )
    print(f"  Verified: PyTorch and ONNX outputs match (max diff "
          f"{np.max(np.abs(torch_output - onnx_output)):.2e}).")


def export_ppo_checkpoint_to_onnx(checkpoint_path, output_dir, override_name=None,
                                   opset_version=17, name_prefix="docking_policy",
                                   loading_message="checkpoint", drift_reminder=False):
    """Load one PPO checkpoint and export its policy network to an ONNX file.

    Shared by export_onnx.py (nodrift, whose policy is fully self-contained)
    and export_drift_policy.py (drift, whose export is network-only; see
    drift_reminder below).

    Args:
        checkpoint_path: Full path to a .zip checkpoint file.
        output_dir: Folder the .onnx file is written into (created if
            missing).
        override_name: Optional exact output filename; auto-named from
            the checkpoint if not given (see build_export_output_name).
        opset_version: ONNX opset to target. 17 is broadly compatible
            with recent onnxruntime releases; raise this if a newer op
            is ever needed, lower it if the target runtime is older.
        name_prefix: Prefix for the auto-generated filename, passed
            through to build_export_output_name.
        loading_message: Text describing what is being loaded, printed
            before the checkpoint path (e.g. "DRIFT checkpoint (network
            component only)" for export_drift_policy.py).
        drift_reminder: True prints a reminder that this export is the
            network only, and that full drift behavior also needs
            cwh_lookahead.py's coast check wired into
            rl_translational_control.py.

    Returns:
        The full path to the written .onnx file.
    """
    import torch  # local import: keeps this module lightweight for path-only callers
    from stable_baselines3 import PPO

    print(f"Loading {loading_message}: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    obs_dim = model.observation_space.shape[0]
    action_dim = model.action_space.shape[0]
    print(f"  Observation dim: {obs_dim}, action dim: {action_dim}")

    onnxable_policy = build_onnxable_policy(model.policy)
    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    output_name = build_export_output_name(checkpoint_path, model, override_name, name_prefix)
    output_path = os.path.join(output_dir, output_name)

    torch.onnx.export(
        onnxable_policy,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,  # older TorchScript exporter; dynamo needs the
                        # separate "onnxscript" package this repo lacks.
    )
    print(f"Wrote: {output_path}")

    verify_onnx_export(output_path, onnxable_policy, dummy_input)

    if drift_reminder:
        print(
            "\nReminder: this file is the NETWORK ONLY. Deploying real drift "
            "behavior also requires cwh_lookahead.py's coast check wired into "
            "rl_translational_control.py, see the module docstring above."
        )

    return output_path


def write_tidy_csv(rows, output_path, data_label="data"):
    """Write a list of flat dicts to a CSV, one row per dict.

    The generic building block behind export_episode_results_csv and
    export_trajectory_steps_csv below. Future ablation scripts whose
    per-episode data doesn't fit either shape can call this directly:
    just build a list of dicts with whatever columns matter.

    Args:
        rows: List of dicts. Dicts do not need matching keys; the header
            is the union of every key seen, and a row missing a key
            writes an empty cell there instead of erroring.
        output_path: Path (str or pathlib.Path) to the .csv file to
            write. Parent directories are created if missing.
        data_label: Short phrase naming what this CSV holds (e.g.
            "episode results", "trajectory steps"), printed so the two
            "Saved data" lines a script writes are easy to tell apart.

    Returns:
        The pathlib.Path written to.
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {data_label} ({len(rows)} rows, {len(fieldnames)} columns):\n{output_path}")
    return output_path


def export_episode_results_csv(results, base_dir, subfolder, filename_prefix,
                                win_outcome_value=None, extra_fields=None):
    """Write one formatted CSV row per episode, across one or more evaluated agents.

    Long format, ready for Seaborn with no pivoting, e.g.
    sns.boxplot(data=df, x="condition", y="fuel_l2"). Pulls whichever of
    "fuels", "fuels_l1", "episode_lengths", "episode_rewards",
    "episode_rewards_weighted", "final_speeds", "elapsed_seconds" a
    result dict happens to carry; a result missing one just skips that
    column for its rows, so drift-only and nodrift-only fields don't
    need special-casing here.

    Args:
        results: List of 1+ result dicts, each shaped like
            evaluate_checkpoint()/evaluate_drift_chain()'s return value:
            must carry "label", "checkpoint", and "all_outcomes" (one
            entry per episode), plus any of the per-episode arrays above
            (same length as "all_outcomes").
        base_dir: The script's BASE_DIR (a pathlib.Path).
        subfolder: Per-script folder name, e.g. "drift_vs_drift_test".
        filename_prefix: Start of the filename; timestamp and ".csv" are
            appended automatically.
        win_outcome_value: If given, adds a boolean "win" column, True
            where an episode's outcome equals this value (e.g. "win" for
            drift results, "direct_dock" for nodrift results). Left out
            entirely if not given, since "win" needs a script-specific
            outcome string this function has no way to guess.
        extra_fields: Optional dict merged into every row as constant
            extra columns, e.g. {"noise_std": 0.1, "lookahead_steps": 5}
            for an ablation axis not already covered above. Overrides
            same-named columns computed here.

    Returns:
        The pathlib.Path the CSV was written to.
    """
    per_episode_keys = {
        "fuels": "fuel_l2",
        "fuels_l1": "fuel_l1",
        "episode_lengths": "episode_length",
        "episode_rewards": "episode_reward",
        "episode_rewards_weighted": "episode_reward_weighted",
        "final_speeds": "final_speed",
        "elapsed_seconds": "elapsed_seconds",
    }

    rows = []
    for result in results:
        outcomes = result.get("all_outcomes", [])
        for i, outcome in enumerate(outcomes):
            row = {
                "condition": result.get("label"),
                "checkpoint": result.get("checkpoint"),
                "episode_idx": i,
                "outcome": outcome,
            }
            if win_outcome_value is not None:
                row["win"] = outcome == win_outcome_value
            for src_key, column in per_episode_keys.items():
                values = result.get(src_key)
                if values is not None and i < len(values):
                    row[column] = values[i]
            if extra_fields:
                row.update(extra_fields)
            rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_dir / "saved_data" / subfolder / f"{filename_prefix}_{timestamp}.csv"
    return write_tidy_csv(rows, output_path, data_label="episode results")


def export_trajectory_steps_csv(results, base_dir, subfolder, filename_prefix, extra_fields=None):
    """Write one formatted CSV row per timestep, for the recorded-trajectory subset.

    Only covers the episodes a result dict kept a full trajectory for
    (result["trajectories"]/["thrusts"]/["speeds"], the first
    NUM_PLOT_EPISODES episodes in every eval script), not the full
    episode count export_episode_results_csv covers. Meant for
    time-series plots, e.g.
    sns.lineplot(data=df, x="step", y="thrust", hue="condition"), which
    averages across episode_idx automatically.

    Args:
        results: List of 1+ result dicts, each carrying "trajectories",
            "thrusts", "speeds" (parallel per-episode lists) and
            "outcomes" (one label per recorded episode).
        base_dir: The script's BASE_DIR (a pathlib.Path).
        subfolder: Per-script folder name, e.g. "drift_vs_drift_test".
        filename_prefix: Start of the filename; "_steps", a timestamp,
            and ".csv" are appended automatically.
        extra_fields: Optional dict merged into every row, see
            export_episode_results_csv.

    Returns:
        The pathlib.Path the CSV was written to.
    """
    import numpy as np

    rows = []
    for result in results:
        trajectories = result.get("trajectories", [])
        thrusts = result.get("thrusts", [])
        speeds = result.get("speeds", [])
        outcomes = result.get("outcomes", [])
        for ep_idx, traj in enumerate(trajectories):
            traj = np.array(traj)
            thrust = thrusts[ep_idx] if ep_idx < len(thrusts) else []
            speed = speeds[ep_idx] if ep_idx < len(speeds) else []
            outcome = outcomes[ep_idx] if ep_idx < len(outcomes) else None
            for step in range(len(traj)):
                row = {
                    "condition": result.get("label"),
                    "checkpoint": result.get("checkpoint"),
                    "episode_idx": ep_idx,
                    "step": step,
                    "x": float(traj[step][0]),
                    "y": float(traj[step][1]),
                    "z": float(traj[step][2]),
                    "distance": float(np.linalg.norm(traj[step])),
                    "thrust": float(thrust[step]) if step < len(thrust) else "",
                    "speed": float(speed[step]) if step < len(speed) else "",
                    "outcome": outcome,
                }
                if extra_fields:
                    row.update(extra_fields)
                rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_dir / "saved_data" / subfolder / f"{filename_prefix}_steps_{timestamp}.csv"
    return write_tidy_csv(rows, output_path, data_label="trajectory steps")


def get_package_versions():
    """Capture installed package versions for a training run's metadata.

    Recorded so a checkpoint's exact dependency state can be checked
    later if reproducibility ever breaks (see drift_initial_trainer.py's
    warning that seeded reproducibility depends on Gymnasium's
    version, which has changed behavior before). No git integration.

    Returns:
        A dict: "python_version", plus one entry per package in
        ("stable_baselines3", "gymnasium", "numpy", "torch") holding its
        installed version string, or None if that package is not
        importable. Never raises.
    """
    import sys

    versions = {"python_version": sys.version.split()[0]}
    for pkg in ("stable_baselines3", "gymnasium", "numpy", "torch"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", None)
        except ImportError:
            versions[pkg] = None
    return versions


# Chief orbit mean motion (rad/s), matching docking_env.py/drift_env.py.
# Used to reproduce the distance-scaled speed limit when sampling starts.
_TEST_SET_MEAN_MOTION = 0.001027


def sample_start_states(num_states, seed, min_pos_bound=100.0,
                        max_pos_bound=150.0, max_vel_bound=0.4):
    """Sample start states usable by both the drift and nodrift agents.

    Mirrors the environments' rejection sampling: uniform position
    and velocity, rejected if too close, too far, or faster than the
    shared distance-scaled speed limit. Bounds default to the tightest
    values both agents train under, so a state is in-distribution for
    either one.

    Args:
        num_states: How many states to draw.
        seed: Seed for the generator, so a set is reproducible.
        min_pos_bound: Minimum starting distance from the chief, meters.
        max_pos_bound: Maximum starting distance from the chief, meters.
        max_vel_bound: Maximum starting speed per axis, m/s.

    Returns:
        A list of 7-element lists: [x, y, z, vx, vy, vz, timestep=0].
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    states = []
    while len(states) < num_states:
        pos = rng.uniform(-max_pos_bound, max_pos_bound, size=3)
        vel = rng.uniform(-max_vel_bound, max_vel_bound, size=3)
        dist = float(np.linalg.norm(pos))
        speed = float(np.linalg.norm(vel))
        if min_pos_bound <= dist <= max_pos_bound:
            if speed <= 0.2 + 2 * _TEST_SET_MEAN_MOTION * dist:
                states.append([float(v) for v in pos] + [float(v) for v in vel] + [0.0])
    return states


def test_set_fingerprint(states):
    """Short hash of a start-state list, for provenance.

    Recorded alongside every result that used the set, so you can catch
    a comparison run against two different sets of starts.

    Args:
        states: List of 7-element start states.

    Returns:
        A 12-character hex string.
    """
    import hashlib

    payload = ";".join(",".join(f"{v:.10g}" for v in s) for s in states)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def save_test_set(path, states, name, description="", bounds=None, seed=None):
    """Write a fixed evaluation start-state set to JSON.

    Args:
        path: Destination .json path. Parent folders are created.
        states: List of 7-element start states.
        name: Short identifier, e.g. "standard_100".
        description: Free text on what the set is for.
        bounds: The sampling bounds dict, recorded for reproducibility.
        seed: The sampling seed, recorded so the set can be regenerated.

    Returns:
        The pathlib.Path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": description,
        "num_states": len(states),
        "generated_seed": seed,
        "bounds": bounds or {},
        "mean_motion": _TEST_SET_MEAN_MOTION,
        "speed_limit": "speed <= 0.2 + 2 * mean_motion * distance",
        "state_format": "[x, y, z, vx, vy, vz, timestep]",
        "fingerprint": test_set_fingerprint(states),
        "states": states,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Saved test set {name!r} ({len(states)} states, "
          f"fingerprint {payload['fingerprint']}):\n{path}")
    return path


def load_test_set(path):
    """Load a fixed evaluation start-state set and verify its fingerprint.

    Args:
        path: Path to a .json file written by save_test_set.

    Returns:
        A dict with "states" as a list of 7-element numpy arrays, plus
        the file's recorded metadata (name, fingerprint, bounds, etc).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the states no longer match the recorded
            fingerprint, meaning the file was edited after being written.
    """
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Test set not found: {path}. Generate one with make_test_set.py, "
            f"or set TEST_SET = None to sample fresh start states each run."
        )
    with open(path) as f:
        payload = json.load(f)

    actual = test_set_fingerprint(payload["states"])
    if actual != payload.get("fingerprint"):
        raise ValueError(
            f"Test set {path} failed its fingerprint check (recorded "
            f"{payload.get('fingerprint')}, computed {actual}). The file was "
            f"edited after it was written, so results using it are not "
            f"comparable to earlier ones."
        )

    payload["states"] = [np.array(s, dtype=float) for s in payload["states"]]
    return payload


def resolve_start_states(test_set_path, num_episodes, fallback_seed,
                         min_pos_bound=100.0, max_pos_bound=150.0,
                         max_vel_bound=0.4, verbose=True, leading_newline=False):
    """Get the start states for a run, from a fixed set or freshly sampled.

    The fixed-set path is what makes two runs comparable: every agent
    faces identical starts, so a metric gap reflects the agents and not
    who drew easier episodes. Falling back to fresh sampling keeps quick
    one-off runs working without a saved set.

    Args:
        test_set_path: Path to a saved set, or None to sample fresh.
        num_episodes: How many states the run needs.
        fallback_seed: Seed used only when sampling fresh.
        min_pos_bound: Sampling bound, fresh sampling only.
        max_pos_bound: Sampling bound, fresh sampling only.
        max_vel_bound: Sampling bound, fresh sampling only.
        verbose: True prints which source was used and its fingerprint.
        leading_newline: True prints a blank line first, to separate this
            line from whatever printed just before it.

    Returns:
        A (states, provenance) tuple. states is a list of num_episodes
        7-element numpy arrays; provenance is a dict recording the
        source, suitable for saving alongside results.

    Raises:
        ValueError: If the fixed set holds fewer states than num_episodes.
    """
    import numpy as np

    if test_set_path is None:
        raw = sample_start_states(num_episodes, fallback_seed, min_pos_bound,
                                  max_pos_bound, max_vel_bound)
        states = [np.array(s, dtype=float) for s in raw]
        provenance = {
            "source": "sampled",
            "seed": fallback_seed,
            "num_states": num_episodes,
            "fingerprint": test_set_fingerprint(raw),
        }
        if verbose:
            if leading_newline:
                print()
            print(f"Start states: sampled fresh from seed {fallback_seed} "
                  f"(set TEST_SET to a saved file for cross-run comparability)")
        return states, provenance

    payload = load_test_set(test_set_path)
    if len(payload["states"]) < num_episodes:
        raise ValueError(
            f"Test set {payload['name']!r} has {len(payload['states'])} states "
            f"but NUM_EPISODES is {num_episodes}. Lower NUM_EPISODES or "
            f"generate a larger set."
        )
    states = payload["states"][:num_episodes]
    provenance = {
        "source": "test_set",
        "name": payload["name"],
        "path": str(test_set_path),
        "fingerprint": payload["fingerprint"],
        "num_states_used": num_episodes,
        "num_states_available": payload["num_states"],
    }
    if verbose:
        if leading_newline:
            print()
        print(f"Start states: test set {payload['name']!r} "
              f"({num_episodes} of {payload['num_states']}, "
              f"fingerprint {payload['fingerprint']})")
    return states, provenance


def wrap_with_action_noise(env, noise_fraction, seed=None):
    """Wrap an environment so every action is scaled by random noise.

    Multiplies each action by a uniform factor in
    [1 - noise_fraction, 1 + noise_fraction] before the environment sees
    it, matching the test-time USE_ACTION_NOISE ablation in the eval
    scripts. Used for TRAIN-time noise, so an agent can be trained under
    the same actuator jitter it is later tested with.

    Because the wrapped environment receives the noisy action, its own
    fuel accounting reflects the noise too, rather than the clean action
    the policy asked for.

    Args:
        env: The environment to wrap.
        noise_fraction: Half-width of the multiplicative noise, so 0.05
            gives the same +/-5% the eval scripts use. 0 disables the
            noise and returns env unwrapped.
        seed: Optional seed for the noise generator, so a training run
            stays reproducible.

    Returns:
        The wrapped environment, or env itself when noise_fraction is 0.
    """
    if not noise_fraction:
        return env

    import gymnasium as gym
    import numpy as np

    class _ActionNoiseWrapper(gym.ActionWrapper):
        """Scales each action by uniform multiplicative noise."""

        def __init__(self, inner_env, fraction, noise_seed):
            super().__init__(inner_env)
            self.noise_fraction = fraction
            self._rng = np.random.default_rng(noise_seed)

        def action(self, action):
            """Scale the action by a fresh uniform noise factor."""
            factor = self._rng.uniform(1.0 - self.noise_fraction,
                                       1.0 + self.noise_fraction,
                                       size=np.shape(action))
            return action * factor

    return _ActionNoiseWrapper(env, noise_fraction, seed)


def cap_blas_threads_per_worker(n=1):
    """Limit NumPy's BLAS backend to n threads, for use before spawning workers.

    All three trainers train with a SubprocVecEnv of N_ENVS worker
    processes. On Windows, multiprocessing can only use the "spawn"
    start method (no fork), so every worker re-imports NumPy fresh in
    its own process. Standard PyPI NumPy wheels link OpenBLAS, which by
    default sizes its internal thread pool to the full logical core
    count, independently in every process. With N_ENVS workers all doing
    that on top of the main process's own PyTorch threads (see
    torch.set_num_threads(), a separate, already-bounded setting), the
    result is real oversubscription: more threads than physical cores,
    all contending for the same CPU, which shows up as high CPU usage
    without training going proportionally faster. Not a
    hypothetical: confirmed via `requirements.txt` pinning a standard
    PyPI numpy build.

    Call this once in the main process, before constructing the
    SubprocVecEnv (make_vec_env/SubprocVecEnv). Spawned children inherit
    the parent's environment variables at spawn time, and OpenBLAS/MKL/
    the reference BLAS all read these at NumPy import time in each
    process, so setting them here (even after this process's own NumPy
    is already imported) still correctly limits every worker process
    spawned afterward. n=1 is the right default: parallelism already
    comes from N_ENVS separate processes, so each process being
    single-BLAS-threaded is the correct configuration, not a compromise.

    Args:
        n: Threads per process for OpenBLAS/MKL/numexpr. 1 unless you
            have very few N_ENVS workers and idle cores to spare, in
            which case raising this (and re-running) may help; there is
            no single right value for every machine, so this is left as
            a parameter rather than hardcoded.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


_kl_message_patched = False


def clarify_ppo_kl_early_stop_message():
    """Make PPO's KL early-stopping print report real numbers.

    Stable-baselines3's PPO.train() prints "Early stopping at step
    {epoch} due to reaching max kl: {approx_kl_div:.2f}" when a
    minibatch's measured KL divergence exceeds 1.5 * target_kl. Read at
    face value this misleads on two points: the printed number is the
    measured divergence, not the configured limit, and the real trigger
    is 1.5 * target_kl, not target_kl itself. Both values differ per
    trainer and per run, so they have to be read off the actual model.

    This patches stable_baselines3.ppo.ppo's own module-level print
    (not the interpreter's builtin, so nothing outside that one module
    is affected) to intercept exactly that message and reprint it with
    the epoch count, the measured KL, and the real 1.5x target_kl
    threshold spelled out. Every other print in the library passes
    through unchanged. Safe to call more than once per trainer script,
    even if imported more than once, with no extra effect after the first.

    Call this once, near the top of a trainer script, before any
    PPO(...) is constructed or PPO.load() is called.
    """
    global _kl_message_patched
    if _kl_message_patched:
        return

    import builtins
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import ppo as _ppo_module

    _real_print = builtins.print
    _original_train = PPO.train
    _training_model = []  # stack; supports nested/re-entrant train() calls

    _EARLY_STOP_RE = re.compile(
        r"^Early stopping at step (\d+) due to reaching max kl: ([\d.eE+-]+)$"
    )

    def _clarified_print(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            match = _EARLY_STOP_RE.match(args[0])
            if match and _training_model:
                epoch = int(match.group(1))
                measured_kl = float(match.group(2))
                model = _training_model[-1]
                n_epochs = getattr(model, "n_epochs", "?")
                target_kl = getattr(model, "target_kl", None)
                threshold = 1.5 * target_kl if target_kl is not None else float("nan")
                _real_print(
                    f"Update stopped early at epoch {epoch} of {n_epochs}: "
                    f"KL divergence reached {measured_kl:.4f} "
                    f"(limit {threshold:.4f}, 1.5x target_kl={target_kl})."
                )
                _real_print(
                    "This limit protects the policy from changing too "
                    "much within a single epoch."
                )
                return
        _real_print(*args, **kwargs)

    def _tracked_train(self):
        _training_model.append(self)
        try:
            return _original_train(self)
        finally:
            _training_model.pop()

    _ppo_module.print = _clarified_print
    PPO.train = _tracked_train
    _kl_message_patched = True
