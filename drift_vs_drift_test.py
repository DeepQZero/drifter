"""Head-to-head drift-chain comparison.

Purpose:
- Compare two checkpoint chains using paired episodes.
- Keep the same seed and start states so differences come from the
  checkpoints.
- Isolate one drift_env.py flag while holding everything else constant.

Related scripts:
- drift_full_test.py: one chain plus extra ablations
- drift_vs_nodrift_test.py: drift vs. no-drift baseline
- nodrift_full_test.py: two no-drift checkpoints

Checkpoint handling:
- CHECKPOINT_A and CHECKPOINT_B each resolve a full 9-stage chain.
- AUTO_LABEL reads both run_metadata.json files to detect the comparison
  type without relying on a handwritten label.
"""

import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import drift_env
from drift_env import DriftTestEnv
from evaluation_utilities import (
    extract_stage,
    rollout_drift_chain_episode,
    resolve_drift_eval_config,
    eval_config_export_fields,
    resolve_obs_flags,
    resolve_drift_model_chain,
    resolve_drift_stage_thresholds,
    resolve_drift_stage_obs_time_norm,
    resolve_env_flags_from_metadata,
    apply_dock_radius_override,
    override_dock_threshold,
    warn_if_win_condition_differs,
    plural_word,
    fuel_statistics,
    auto_label_drift_checkpoints,
    mean_pairwise_angle_deg,
    describe_comparison_type,
    print_paired_comparison_table,
    plot_paired_approach_directions,
    plot_paired_trajectories_3d,
    plot_paired_diagnostics,
    describe_drift_chain,
    resolve_and_print_seed,
    export_episode_results_csv,
    export_trajectory_steps_csv,
    detect_agent_mode,
    resolve_start_states,
    SPACE_CONTROLS_FLAG_KEYS,
    DRIFT_REWARD_FLAG_KEYS,
    COAST_THRUST_THRESHOLD,
    TRUE_COAST_THRESHOLD,
    ACTION_SATURATION_THRESHOLD,
    APPROACH_CHECK_DISTANCE,
    MAX_STEPS_PER_EPISODE,
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"

# --- Settings ---
# Grouped by what they affect; execution starts below.

# Checkpoints:

# The two chains to compare. Each resolves the newest checkpoint per stage
# in its run folder, same convention as drift_full_test.py.
CHECKPOINT_A = "latest"
CHECKPOINT_B = "run:safe_PPO_30"

# Limit on models per chain, kept high so every stage is picked up.
NUM_MODELS = 100

# Run setup:

# Number of paired episodes to evaluate per chain.
NUM_EPISODES = 30

# True:  exact closed-form CWH propagation instead of solve_ivp. Matches
#        what the trainers use, and roughly 8x faster; agrees with
#        solve_ivp to about 1e-11 m, far below any dock threshold.
# False: solve_ivp (RK45), the same integrator the environment used to
#        default to. Use this to check a result against the
#        original solver.
FAST_ANALYTIC_PROPAGATION = True
drift_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION

# Fixed start states so every run faces identical episodes: 100 states at
# 100-150 m, 0.4 m/s per axis. NUM_EPISODES picks how many are used.
# None: sample fresh from SEED instead (comparable only within that run).
TEST_SET = BASE_DIR / "test_sets" / "standard_100.json"

# Seed for paired episodes; both chains use identical start states.
# None:   pick a random seed, printed at startup.
# An int: fix to a specific comparison.
SEED = None  # example: 12345

# True:  pop up a window for each plot.
# False: save PNGs to saved_figures/, no window (auto-skipped under a
#        headless Matplotlib backend).
SHOW_PLOTS = True

# Ablations:

# Multiply every action by random noise in [0.95, 1.05], a robustness test.
# True:  add noise.
# False: use each model's exact action.
USE_ACTION_NOISE = False

# Scale the observation's timestep element by each stage's own trained
# max_episode_len instead of the eval env's (9,000), which constrains obs[6]
# near zero all flight.
# True:  per-stage divisor, matching what each policy trained on.
# False: the eval env's own max_episode_len.
USE_PER_STAGE_OBS_TIME_NORM = True

# Win condition overrides:
# Force either chain's final stage to dock against a distance/speed it
# did not train with. None: use each chain's own trained thresholds,
# unchanged. See evaluation_utilities.override_dock_threshold.
A_DOCKING_POSITION_OVERRIDE = None  # meters, e.g. 0.5. None: use chain A's own trained value.
A_DOCKING_SPEED_OVERRIDE = None     # m/s, e.g. 0.2. None: use chain A's own trained value.
B_DOCKING_POSITION_OVERRIDE = None  # meters, e.g. 0.5. None: use chain B's own trained value.
B_DOCKING_SPEED_OVERRIDE = None     # m/s, e.g. 0.2. None: use chain B's own trained value.

# Labels:

# Where the plot labels come from.
# True:  auto-detect the comparison type by diffing run_metadata.json.
# False: always use the manual LABEL_A/LABEL_B below.
AUTO_LABEL = True

# Manual labels, used when AUTO_LABEL is False, or as the fallback when
# auto-detection can't run (a chain has no run_metadata.json).
LABEL_A = "Drift A"
LABEL_B = "Drift B"

# Console output:

# Prints a "Model N/9 (stage S): WIN/FAIL" line for every stage switch.
# True:  print all of them.
# False: only the per-chain win line and final table.
PRINT_STAGE_RESULTS = False

# Saved output:

# Formatted per-episode and per-step CSVs under saved_data/, ready for Seaborn.
# True:  write them (see evaluation_utilities.export_episode_results_csv).
# False: skip them.
EXPORT_RESULTS_CSV = True

# Combined 3D trajectory plot with both chains overlaid.
# True:  save it.
# False: skip it.
PLOT_TRAJECTORIES = True

# How many of the first episodes per chain to record and plot. Kept below
# NUM_EPISODES so the combined plot stays readable.
NUM_PLOT_EPISODES = 10

# 2x2 per-step diagnostics figure: thrust, distance, and speed vs. time.
# True:  save it.
# False: skip it.
PLOT_DIAGNOSTICS = True

# First N episodes per chain to overlay in the diagnostics figure.
DIAG_NUM_EPISODES = 10

# Scatter plot of each episode's approach direction.
# True:  save it.
# False: skip it.
PLOT_APPROACH_DIRECTIONS = False

# Figure style:

# Colors for A/B everywhere (Okabe-Ito orange/teal pair).
# When AUTO_LABEL detects "seed_only", __main__ swaps to two shades of blue
# so the plot doesn't imply a real difference.
PLOT_COLORS = {"A": "#E69F00", "B": "#009E73"}
SEED_ONLY_PLOT_COLORS = {"A": "#4C72B0", "B": "#8DA0CB"}

# Color each trajectory by thrust magnitude (bright = thrust, dark = coast).
# True:  color by thrust.
# False: flat per-chain color.
COLOR_TRAJECTORY_BY_THRUST = True

# Scale line width by thrust and mark any step above THRUST_BURST_FRACTION
# of the episode's max, so brief hard thrusts stand out.
# True:  show burst markers.
# False: plain lines.
THRUST_BURST_MARKERS = True

# How much those burst markers pop.
# True:  one bright highlight color (BRIGHT_THRUST_COLOR), enlarged.
# False: colored by the thrust colormap, subtler.
BRIGHT_THRUST_HIGHLIGHTS = False
BRIGHT_THRUST_COLOR = "#FF2A00"

# Fraction of max thrust above which a step gets a burst marker.
THRUST_BURST_FRACTION = 0.6

# Colormap for COLOR_TRAJECTORY_BY_THRUST. All but "grey_to_warm" are
# colorblind-safe.
#   "discrete"           Default. Grey (coast) to navy (thrust), 0.25 N bands.
#   "cividis"/"viridis"  Continuous, perceptually uniform.
#   "grey_to_warm"       High contrast, NOT colorblind-safe.
THRUST_COLORMAP = "discrete"

# Where the 3D plot's thrust colorbar sits.
# True:  horizontal bar below the axes.
# False: vertical bar at the right (Matplotlib's default).
THRUST_COLORBAR_AT_BOTTOM = False

# Metric thresholds (COAST_THRUST_THRESHOLD, TRUE_COAST_THRESHOLD,
# ACTION_SATURATION_THRESHOLD, APPROACH_CHECK_DISTANCE) are imported from
# evaluation_utilities to ensure consistent comparison across scripts.
# Figure-style settings stay per-script.


def evaluate_drift_chain(model_paths: list, thresholds: list, label: str, seed: int,
                         start_states: list, stage_obs_time_norms: list = None,
                         pos_override: float = None, speed_override: float = None) -> dict:
    """Run NUM_EPISODES paired episodes chaining one drift model curriculum.

    Starts on the hardest model, switches to the next easier one as each
    stage docks; only the final stage's result counts as a win or loss.

    Args:
        model_paths: Chain of checkpoint paths, hardest (index 0) to
            easiest (last index).
        thresholds: (pos_thresh, speed_thresh) per model, same order.
        label: Short display name for this chain in printed output.
        seed: Base seed for the paired episodes (episode i uses seed + i).
        start_states: One start state per episode (see
            evaluation_utilities.resolve_start_states), injected so both
            chains face identical episodes.
        stage_obs_time_norms: Per USE_PER_STAGE_OBS_TIME_NORM, same order
            as model_paths. None leaves the eval env's own obs[6] divisor
            untouched.
        pos_override: Distance, in meters, to require for this chain's
            final stage instead of its own trained pos_thresh. None
            (default) leaves it unchanged. See
            evaluation_utilities.override_dock_threshold.
        speed_override: Speed, in m/s, to require for this chain's final
            stage instead of its own trained speed_thresh. None leaves
            it unchanged.

    Returns:
        A dict of aggregate results plus raw trajectories/outcomes for
        the first NUM_PLOT_EPISODES episodes, for plotting.
    """
    models = [PPO.load(str(p), device="cpu") for p in model_paths]
    # Match the eval env to training config. Obs-extending flags come from
    # observation size; reward-side flags come from run_metadata.json.
    # space_controls_obs is read first since it changes the size-to-flags mapping.
    space_controls_flags = resolve_env_flags_from_metadata(
        str(model_paths[0]), keys=SPACE_CONTROLS_FLAG_KEYS
    )
    saferl_obs, custom_braking_margin_obs = resolve_obs_flags(
        models[0].observation_space.shape[0],
        space_controls_obs=space_controls_flags.get("space_controls_obs", False),
    )
    drift_env_flags = {
        "saferl_obs": saferl_obs,
        "custom_braking_margin_obs": custom_braking_margin_obs,
        **resolve_env_flags_from_metadata(
            str(model_paths[0]),
            keys=DRIFT_REWARD_FLAG_KEYS,
        ),
        **space_controls_flags,
    }
    # Held until after the chain header prints below, so the flags line
    # stays attached to the chain it describes.
    non_default_flags = {k: v for k, v in drift_env_flags.items() if v}

    wins = 0
    fuels, fuels_l1, timesteps, episode_rewards = [], [], [], []
    elapsed_seconds_list = []  # real simulated time across the whole chain, per episode
    failure_counts = {
        "crash": 0, "unsafe": 0, "out_of_bounds": 0, "fuel": 0,
        "timeout": 0, "unknown": 0,
    }
    trajectories, thrusts, speeds, outcomes = [], [], [], []
    all_outcomes = []  # win/loss for every episode, not just the recorded subset
    coast_steps = 0
    true_coast_steps = 0
    saturated_steps = 0
    violation_steps = 0  # steps exceeding env.env.velocity_limit(), DRL for Space Controls (2024)'s docking metric
    # Per-episode violation rate, unweighted average across episodes.
    # violation_fraction above is step-weighted; this gives every episode equal say.
    episode_violation_fractions = []
    total_steps = 0
    entry_directions = []
    final_speeds = []  # speed at episode end, DRL for Space Controls (2024)'s other docking-specific metric

    print()
    print(f"{label}: chain of {len(models)} models, hardest {os.path.basename(model_paths[0])}")
    print("  Environment flags:")
    if non_default_flags:
        for flag_name, flag_value in sorted(non_default_flags.items()):
            print(f"    {flag_name}: {flag_value}")
    else:
        print("    (all default/off)")
    if pos_override is not None or speed_override is not None:
        _trained_pos = apply_dock_radius_override(thresholds[-1][0], space_controls_flags, "drift")
        override_dock_threshold(
            _trained_pos, thresholds[-1][1], pos_override, speed_override, label=label)
    for ep in range(NUM_EPISODES):
        # verbose=False: the resolved config already printed once, before
        # this loop, via the module-level drift_test_config resolution below.
        curriculum, _ = resolve_drift_eval_config(verbose=False)
        env = DriftTestEnv(**drift_env_flags, **curriculum)
        # Override the env's own sampling so both chains see the same
        # start for a given episode index.
        env.env.fixed_start = True
        env.env.fixed_state = start_states[ep].copy()
        obs, _ = env.reset(seed=seed + ep)

        ep_result = rollout_drift_chain_episode(
            env, obs, models, thresholds, model_paths,
            max_steps=MAX_STEPS_PER_EPISODE,
            use_action_noise=USE_ACTION_NOISE,
            print_stage_results=PRINT_STAGE_RESULTS,
            stage_obs_time_norms=stage_obs_time_norms,
            final_pos_override=pos_override,
            final_speed_override=speed_override,
        )

        step_count = ep_result["episode_length"]
        outcomes_this_episode = ep_result["outcome"]
        if ep_result["win"]:
            wins += 1
        else:
            failure_counts[ep_result["failure_type"]] += 1

        # Per-step stats derived from the trace. Index 0 is the initial
        # (pre-episode) placeholder, so real steps start at index 1.
        episode_violation_steps = 0
        entry_recorded = False
        for j in range(1, step_count + 1):
            thrust_mag = ep_result["thrusts"][j]
            total_steps += 1
            if thrust_mag < COAST_THRUST_THRESHOLD:
                coast_steps += 1
            if thrust_mag < TRUE_COAST_THRESHOLD:
                true_coast_steps += 1
            if np.max(np.abs(ep_result["actions"][j])) > ACTION_SATURATION_THRESHOLD:
                saturated_steps += 1

            position = ep_result["positions"][j]
            distance = float(np.linalg.norm(position))
            current_speed = ep_result["speeds"][j]
            if current_speed > ep_result["speed_limits"][j]:
                violation_steps += 1
                episode_violation_steps += 1

            if not entry_recorded and distance <= APPROACH_CHECK_DISTANCE and distance > 0:
                entry_directions.append(position.copy() / distance)
                entry_recorded = True

        timesteps.append(step_count)
        elapsed_seconds_list.append(ep_result["elapsed_seconds"])
        episode_rewards.append(ep_result["episode_reward"])
        final_speeds.append(ep_result["speeds"][-1])
        fuels.append(ep_result["fuel"])
        fuels_l1.append(ep_result["fuel_l1"])
        all_outcomes.append(outcomes_this_episode)
        episode_violation_fractions.append(
            episode_violation_steps / step_count if step_count else 0.0)

        # Read position and velocity from the environment state.
        # Checkpoint settings may scale the observation.
        if (PLOT_TRAJECTORIES or PLOT_DIAGNOSTICS) and ep < NUM_PLOT_EPISODES:
            trajectories.append(ep_result["positions"])
            thrusts.append(ep_result["thrusts"])
            speeds.append(ep_result["speeds"])
            outcomes.append(outcomes_this_episode)

    print(f"  Wins: {wins}/{NUM_EPISODES} ({100 * wins / NUM_EPISODES:.1f}%)")

    return {
        "label": label,
        "checkpoint": describe_drift_chain(model_paths),
        "wins": wins,
        "total": NUM_EPISODES,
        "win_rate": wins / NUM_EPISODES,
        **fuel_statistics(fuels, fuels_l1),
        "fuels": fuels,
        "fuels_l1": fuels_l1,
        "all_outcomes": all_outcomes,
        # Per-episode step count and reward, all NUM_EPISODES. Kept raw
        # so export_episode_results_csv can write one row per episode.
        "episode_lengths": timesteps,
        "episode_rewards": episode_rewards,
        "mean_timesteps": float(np.mean(timesteps)),
        # Real simulated seconds and not step count: a coast step covers
        # drift_step_len seconds and an active step covers step_len, so
        # raw step counts undercount a coast-heavy episode.
        "elapsed_seconds": elapsed_seconds_list,
        "mean_time_to_dock": float(np.mean(elapsed_seconds_list)),
        "median_time_to_dock": float(np.median(elapsed_seconds_list)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "median_episode_reward": float(np.median(episode_rewards)),
        "coast_fraction": coast_steps / total_steps if total_steps else 0.0,
        "true_coast_fraction": true_coast_steps / total_steps if total_steps else 0.0,
        "saturation_fraction": saturated_steps / total_steps if total_steps else 0.0,
        # DRL for Space Controls (2024) (arXiv 2405.12355) reports these two metrics
        # directly (Table II); tracked here for a like-for-like comparison.
        "violation_fraction": violation_steps / total_steps if total_steps else 0.0,
        "violation_fraction_episode_mean": (
            float(np.mean(episode_violation_fractions))
            if episode_violation_fractions else 0.0),
        "final_speeds": final_speeds,
        "mean_final_speed": float(np.mean(final_speeds)) if final_speeds else 0.0,
        "approach_spread_deg": mean_pairwise_angle_deg(entry_directions),
        "entry_directions": entry_directions,
        "failure_counts": failure_counts,
        "trajectories": trajectories,
        "thrusts": thrusts,
        "speeds": speeds,
        "outcomes": outcomes,
    }


def plot_comparison(result_a: dict, result_b: dict, seed: int = None, show: bool = True) -> None:
    """Save a 3D trajectory plot with both chains' episodes overlaid.

    Thin wrapper over evaluation_utilities.plot_paired_trajectories_3d(),
    which owns the plot itself so all three paired-comparison scripts
    render it identically. Only this script's colors, thrust-drawing
    flags, and win label ("win") differ.

    Args:
        result_a: Return value of evaluate_drift_chain() for chain A.
        result_b: Same, for chain B.
        seed: The resolved seed, noted in the plot title.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    plot_paired_trajectories_3d(
        result_a, result_b,
        colors=PLOT_COLORS,
        base_dir=BASE_DIR,
        subfolder="drift_vs_drift_test",
        filename_prefix="drift_vs_drift_comparison",
        title=f"Drift vs. Drift Comparison: {result_a['label']} vs {result_b['label']}",
        win_outcome_value="win",
        color_by_thrust=COLOR_TRAJECTORY_BY_THRUST,
        thrust_colormap=THRUST_COLORMAP,
        thrust_burst_markers=THRUST_BURST_MARKERS,
        bright_thrust_highlights=BRIGHT_THRUST_HIGHLIGHTS,
        bright_thrust_color=BRIGHT_THRUST_COLOR,
        thrust_burst_fraction=THRUST_BURST_FRACTION,
        colorbar_at_bottom=THRUST_COLORBAR_AT_BOTTOM,
        seed=seed,
        show=show,
    )


def plot_diagnostics(result_a: dict, result_b: dict, seed: int = None, show: bool = True) -> None:
    """Save a 2x2 per-step diagnostics figure with both chains overlaid.

    Thin wrapper over evaluation_utilities.plot_paired_diagnostics(),
    which owns the plot itself so all three paired-comparison scripts
    render it identically. Only this script's colors and output paths
    differ.

    Args:
        result_a: Return value of evaluate_drift_chain() for chain A.
        result_b: Same, for chain B.
        seed: The resolved seed, noted in the plot title.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    plot_paired_diagnostics(
        result_a, result_b,
        colors=PLOT_COLORS,
        base_dir=BASE_DIR,
        subfolder="drift_vs_drift_test",
        filename_prefix="drift_vs_drift_diagnostics",
        coast_threshold=COAST_THRUST_THRESHOLD,
        diag_num_episodes=DIAG_NUM_EPISODES,
        seed=seed,
        show=show,
    )


def plot_approach_directions(result_a: dict, result_b: dict, seed: int = None,
                             show: bool = True) -> None:
    """Save a scatter plot of each episode's approach direction.

    Thin wrapper over
    evaluation_utilities.plot_paired_approach_directions(), which owns the
    plot itself so all three paired-comparison scripts render it
    identically. Only this script's colors and output paths differ.

    Args:
        result_a: Result dict for side A.
        result_b: Result dict for side B.
        seed: The resolved seed, noted in the plot title.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    plot_paired_approach_directions(
        result_a, result_b,
        colors=PLOT_COLORS,
        base_dir=BASE_DIR,
        subfolder="drift_vs_drift_test",
        filename_prefix="drift_vs_drift_approach_directions",
        check_distance=APPROACH_CHECK_DISTANCE,
        seed=seed,
        show=show,
    )


def print_comparison_table(result_a: dict, result_b: dict, seed: int) -> None:
    """Print the side-by-side comparison table for this script's two results.

    Thin wrapper over evaluation_utilities.print_paired_comparison_table(),
    which keeps all three paired-comparison scripts identical in format.

    Args:
        result_a: Result dict for side A.
        result_b: Result dict for side B.
        seed: The resolved seed the paired episodes ran with.
    """
    print_paired_comparison_table(result_a, result_b, seed, NUM_EPISODES)


if __name__ == "__main__":
    resolved_seed = resolve_and_print_seed(SEED, leading_newline=True)

    model_paths_a = resolve_drift_model_chain(CHECKPOINT_A, CHECKPOINT_ROOT, NUM_MODELS)
    model_paths_b = resolve_drift_model_chain(CHECKPOINT_B, CHECKPOINT_ROOT, NUM_MODELS)

    # Catch a non-drift checkpoint early so the script stops with a clear
    # error before running tests. This checks every stage in the model chain,
    # not just the last one, to catch folder typos or mismatched checkpoints.
    for label, paths in (("A", model_paths_a), ("B", model_paths_b)):
        wrong_family = [p for p in paths if detect_agent_mode(str(p)) != "drift"]
        if wrong_family:
            raise ValueError(
                f"CHECKPOINT_{label} resolved to {len(wrong_family)} "
                f"checkpoint(s) not detected as drift: "
                f"{[str(p) for p in wrong_family]}. This script only "
                f"supports drift checkpoints."
            )

    thresholds_a = [resolve_drift_stage_thresholds(str(p)) for p in model_paths_a]
    thresholds_b = [resolve_drift_stage_thresholds(str(p)) for p in model_paths_b]
    obs_time_norms_a = (
        [resolve_drift_stage_obs_time_norm(str(p)) for p in model_paths_a]
        if USE_PER_STAGE_OBS_TIME_NORM else None)
    obs_time_norms_b = (
        [resolve_drift_stage_obs_time_norm(str(p)) for p in model_paths_b]
        if USE_PER_STAGE_OBS_TIME_NORM else None)

    # Identify which run each side resolved to before evaluating either.
    run_folder_a = model_paths_a[0].parent.name
    run_folder_b = model_paths_b[0].parent.name
    print(f"A: {CHECKPOINT_A} -> {run_folder_a} ({len(model_paths_a)} stages)")
    for path, (pos_thresh, speed_thresh) in zip(model_paths_a, thresholds_a):
        print(f"  stage {extract_stage(str(path)):<3} {path.name}  "
              f"(pos_thresh={pos_thresh}, speed_thresh={speed_thresh})")
    print(f"B: {CHECKPOINT_B} -> {run_folder_b} ({len(model_paths_b)} stages)")
    for path, (pos_thresh, speed_thresh) in zip(model_paths_b, thresholds_b):
        print(f"  stage {extract_stage(str(path)):<3} {path.name}  "
              f"(pos_thresh={pos_thresh}, speed_thresh={speed_thresh})")
    print()

    # Auto-detect the comparison type instead of trusting hand-written
    # LABEL_A/LABEL_B, which becomes outdated once CHECKPOINT_A/B change.
    if AUTO_LABEL:
        detected = auto_label_drift_checkpoints(
            str(model_paths_a[0]), str(model_paths_b[0]),
            default_label_a=LABEL_A, default_label_b=LABEL_B,
        )
        label_a, label_b = detected["label_a"], detected["label_b"]
        print(f"Comparison type (auto-detected): {detected['comparison_type']}")
        explanation_lines = describe_comparison_type(detected["comparison_type"], detected["differences"])
        print(f"  What this means: {explanation_lines[0]}")
        for line in explanation_lines[1:]:
            print(f"  {line}")
        if detected["differences"]:
            print("Differences found:")
            for field, (val_a, val_b) in detected["differences"].items():
                print(f"  {field}: A={val_a}  B={val_b}")
        else:
            print("No differences found between the two chains' recorded config.")
        if detected["note"]:
            print("Notes:")
            for note_line in detected["note"].splitlines():
                print(f"  {note_line}")
        if detected["comparison_type"] == "seed_only":
            # Recolor to same-hue shades so the plot doesn't imply a real A-vs-B difference.
            PLOT_COLORS = SEED_ONLY_PLOT_COLORS
    else:
        label_a, label_b = LABEL_A, LABEL_B

    print()
    print(f"{plural_word(NUM_EPISODES, 'Episode')} per chain: {NUM_EPISODES}")

    # Each chain's win condition is its final (tightest) stage, thresholds_*[-1].
    # apply_dock_radius_override() corrects for space_controls_dock_radius,
    # which replaces pos_thresh in the env constructor without updating the recorded config.
    _pos_a, _speed_a = thresholds_a[-1]
    _pos_b, _speed_b = thresholds_b[-1]
    _flags_a = resolve_env_flags_from_metadata(str(model_paths_a[-1]), keys=("space_controls_dock_radius",))
    _flags_b = resolve_env_flags_from_metadata(str(model_paths_b[-1]), keys=("space_controls_dock_radius",))
    # Layer A_DOCKING_POSITION_OVERRIDE/B_DOCKING_POSITION_OVERRIDE on top so this warning reflects
    # the win condition each chain evaluates under, not the trained
    # value an override has already changed.
    _effective_pos_a, _effective_speed_a = override_dock_threshold(
        apply_dock_radius_override(_pos_a, _flags_a, "drift"), _speed_a,
        A_DOCKING_POSITION_OVERRIDE, A_DOCKING_SPEED_OVERRIDE, verbose=False,
    )
    _effective_pos_b, _effective_speed_b = override_dock_threshold(
        apply_dock_radius_override(_pos_b, _flags_b, "drift"), _speed_b,
        B_DOCKING_POSITION_OVERRIDE, B_DOCKING_SPEED_OVERRIDE, verbose=False,
    )
    warn_if_win_condition_differs(
        label_a, _effective_pos_a, _effective_speed_a,
        label_b, _effective_pos_b, _effective_speed_b,
    )

    # resolve_drift_eval_config() is the drift test env config both chains
    # evaluate under.
    drift_test_config, _ = resolve_drift_eval_config()
    start_states, start_state_provenance = resolve_start_states(
        TEST_SET, NUM_EPISODES, resolved_seed,
        min_pos_bound=drift_test_config['min_init_pos_bound'],
        max_pos_bound=drift_test_config['max_init_pos_bound'],
        max_vel_bound=drift_test_config['max_init_vel_bound'],
    )
    result_a = evaluate_drift_chain(model_paths_a, thresholds_a, label_a,
                                    resolved_seed, start_states, obs_time_norms_a,
                                    pos_override=A_DOCKING_POSITION_OVERRIDE, speed_override=A_DOCKING_SPEED_OVERRIDE)
    result_b = evaluate_drift_chain(model_paths_b, thresholds_b, label_b,
                                    resolved_seed, start_states, obs_time_norms_b,
                                    pos_override=B_DOCKING_POSITION_OVERRIDE, speed_override=B_DOCKING_SPEED_OVERRIDE)

    if EXPORT_RESULTS_CSV:
        # Records the drift eval config (stage 9, absent from checkpoint
        # metadata) so the CSV is self-describing. start_state_provenance
        # is included because a loaded test set supplies its own bounds.
        _eval_fields = eval_config_export_fields(drift_test_config, start_state_provenance)
        export_episode_results_csv([result_a, result_b], BASE_DIR, "drift_vs_drift_test",
                                    "drift_vs_drift_episodes", win_outcome_value="win",
                                    extra_fields=_eval_fields)
        print()
        export_trajectory_steps_csv([result_a, result_b], BASE_DIR, "drift_vs_drift_test",
                                     "drift_vs_drift_trajectories", extra_fields=_eval_fields)

    print_comparison_table(result_a, result_b, resolved_seed)

    if PLOT_TRAJECTORIES:
        plot_comparison(result_a, result_b, seed=resolved_seed, show=SHOW_PLOTS)
    if PLOT_DIAGNOSTICS and (result_a["trajectories"] or result_b["trajectories"]):
        plot_diagnostics(result_a, result_b, seed=resolved_seed, show=SHOW_PLOTS)
    if PLOT_APPROACH_DIRECTIONS and (result_a["entry_directions"] or result_b["entry_directions"]):
        plot_approach_directions(result_a, result_b, seed=resolved_seed, show=SHOW_PLOTS)
