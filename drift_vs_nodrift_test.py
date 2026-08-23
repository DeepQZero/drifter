"""Head-to-head comparison of two agents:

- Drift agent: a chained curriculum of models in DriftTestEnv
- Nodrift agent: one checkpoint in SpaceCraftDockingEnv3D

Paired episodes:
- Each episode starts from the same shared state
- That start state is created once and reused for both agents
- This keeps the comparison fair and avoids easier starts for one side

Full graphics view:
- Reuses the same plotting style as nodrift_full_test.py
- Shows 3D thrust-colored paths, per-step diagnostics, and approach spread
- Agent A is always the drift chain; Agent B is always the nodrift model

Related scripts:
- drift_full_test.py: drift chain only, with extra ablation checks
- nodrift_full_test.py: paired comparison between two nodrift checkpoints
- checkpoint_test.py: one checkpoint, one stage, multiple episodes

Checkpoint setup:
- DRIFT_CHECKPOINT and DRIFT_NUM_MODELS resolve the drift chain like drift_full_test.py
- The safe_PPO folder prefix helps keep "latest" from picking a nodrift run
- NODRIFT_CHECKPOINT resolves a single checkpoint like nodrift_full_test.py
- SEED is random by default and printed at startup for reproducibility
"""

import contextlib
import io
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import drift_env
import docking_env
from drift_env import DriftTestEnv
from docking_env import SpaceCraftDockingEnv3D
from evaluation_utilities import (
    extract_stage,
    classify_failure_nodrift,
    rollout_drift_chain_episode,
    resolve_drift_eval_config,
    eval_config_export_fields,
    resolve_checkpoint_paths,
    resolve_obs_flags,
    resolve_nodrift_env_config,
    resolve_drift_stage_thresholds,
    resolve_drift_stage_obs_time_norm,
    apply_dock_radius_override,
    override_dock_threshold,
    warn_if_win_condition_differs,
    save_and_show_figure,
    plural_word,
    build_thrust_colormap,
    max_thrust_across_results,
    fuel_statistics,
    resolve_env_flags_from_metadata,
    mean_pairwise_angle_deg,
    describe_drift_chain,
    print_paired_comparison_table,
    plot_paired_approach_directions,
    resolve_and_print_seed,
    export_episode_results_csv,
    export_trajectory_steps_csv,
    detect_agent_mode,
    resolve_start_states,
    new_safe_action_counts,
    summarize_safe_action_counts,
    set_equal_3d_axes,
    SPACE_CONTROLS_FLAG_KEYS,
    NODRIFT_REWARD_FLAG_KEYS,
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

# Drift checkpoint chain: "latest", "run:safe_PPO_20", a glob pattern
# like "run:my_model_*", or an exact checkpoint folder path.
DRIFT_CHECKPOINT = "latest"

# How many models to include in the drift chain. Keep this high enough to
# include every stage in the chain when using a grouped run history.
DRIFT_NUM_MODELS = 100

# Single nodrift checkpoint. Use one of the same formats as above.
# "latest" means the newest standalone no-drift model.
# For a curriculum baseline, try a value such as
# "run:nodrift_curriculum_PPO_<N>".
NODRIFT_CHECKPOINT = "latest"

# Fallback environment settings for the no-drift agent. These are only used
# if the checkpoint does not include run_metadata.json.
# Drift agents do not need this fallback because get_curriculum(9) provides
# the environment settings for them.
FALLBACK_ENV_CONFIG = {
    "min_init_pos_bound": 100,
    "max_init_pos_bound": 150,
    "max_init_vel_bound": 0.4,
    "max_boundary_box": 200,
    "max_episode_len": 2000,
    "pos_thresh": 0.5,
    "speed_thresh": 0.2,
    "max_total_dv": 2500,
}

# Run setup:

# Number of paired episodes to evaluate per agent.
NUM_EPISODES = 50

# True:  exact closed-form CWH propagation instead of solve_ivp, for both
#        agents. Matches what the trainers use, and roughly 8x faster;
#        agrees with solve_ivp to about 1e-11 m, far below any dock
#        threshold.
# False: solve_ivp (RK45), the same integrator the environments used to
#        default to. Use this to check a result against the
#        original solver.
FAST_ANALYTIC_PROPAGATION = True
drift_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION
docking_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION

# Fixed start states so every run faces identical episodes (100 states,
# 100-150m, 0.4 m/s per axis). NUM_EPISODES picks how many are used.
# None: sample fresh from SEED instead.
TEST_SET = BASE_DIR / "test_sets" / "standard_100.json"

# Seed for the run, only used to sample starts when TEST_SET is None.
# None:   pick a random seed, printed at startup so the run is reproducible.
# An int: fix to a specific comparison.
SEED = None  # example: 12345

# True:  pop up a window for each plot.
# False: save each plot's PNG to saved_figures/, skip the window.
# Also skipped automatically under a headless backend (e.g. MPLBACKEND=Agg).
SHOW_PLOTS = True

# Ablations:
# Both apply to the drift agent only; the nodrift checkpoint always uses
# its raw action.

# Multiply the drift agent's actions by random noise in [0.95, 1.05],
# matching drift_full_test.py's default.
# True:  add noise.
# False: use the model's exact action.
USE_ACTION_NOISE = False

# Replace the drift agent's action with a one-step lookahead safety check
# (see get_safe_action), matching drift_full_test.py's ablation.
# True:  use the safety-checked action.
# False: use the model's raw action.
USE_SAFE_ACTION = False

# Scale the observation's timestep element by each stage's trained
# max_episode_len instead of the eval env's (9,000), which constrains obs[6]
# near zero all flight.
# True:  per-stage divisor, matching what each policy trained on.
# False: the eval env's own max_episode_len.
USE_PER_STAGE_OBS_TIME_NORM = True

# Win condition overrides:
# Force one or both sides to dock against a distance/speed they were not
# trained with, e.g. testing a checkpoint trained at a looser Space
# Controls dock radius against the stricter target its comparison
# partner uses, or the reverse. 
# None: use each checkpoint's own trained thresholds, unchanged. 
# See evaluation_utilities.override_dock_threshold.
DRIFT_DOCKING_POSITION_OVERRIDE = None    # meters, e.g. 0.5. None: use drift's own trained value.
DRIFT_DOCKING_SPEED_OVERRIDE = None       # m/s, e.g. 0.2. None: use drift's own trained value.
NODRIFT_DOCKING_POSITION_OVERRIDE = None  # meters, e.g. 10.0. None: use nodrift's own trained value.
NODRIFT_DOCKING_SPEED_OVERRIDE = None     # m/s, e.g. 0.2. None: use nodrift's own trained value.

# Labels:

LABEL_A = "Drift"
LABEL_B = "Nodrift"

# Adds how the nodrift checkpoint was trained to its label. Unknown folder names keep the plain label.
# True:  "(standalone)" for nodrift_PPO_<N>, "(curriculum)" for
#        nodrift_curriculum_PPO_<N>.
# False: always use LABEL_A/LABEL_B as written.
SHOW_NODRIFT_TRAINING_STYLE = True

# Console output:

# A "Model N/9 (stage S): WIN/FAIL" line for every stage switch in every episode.
# True:  print all of them.
# False: only the final summary.
PRINT_STAGE_RESULTS = False

# Saved output:

# Formatted per-episode and per-step CSVs under saved_data/, ready for Seaborn.
# True:  write them (see evaluation_utilities.export_episode_results_csv).
# False: skip them.
EXPORT_RESULTS_CSV = True

# Combined 3D trajectory plot with both agents overlaid.
# True:  save it.
# False: skip it.
PLOT_TRAJECTORIES = True

# How many of the first episodes per agent to record and plot. Kept below
# NUM_EPISODES so the combined plot stays readable.
NUM_PLOT_EPISODES = 50

# 2x2 per-step diagnostics figure (thrust, distance, speed vs. time).
# An efficient agent shows long near-zero-thrust coasts with short braking thrusts.
# True:  save it.
# False: skip it.
PLOT_DIAGNOSTICS = True

# First N episodes per agent to overlay in the diagnostics figure.
DIAG_NUM_EPISODES = 50

# Scatter plot of each episode's approach direction at APPROACH_CHECK_DISTANCE.
# A converged approach clusters tightly; a diverse one spreads across the sphere.
# True:  save it.
# False: skip it.
PLOT_APPROACH_DIRECTIONS = False

# Bar chart of per-episode fuel (Delta-v L1), drift vs nodrift, across all NUM_EPISODES.
# Not limited to NUM_PLOT_EPISODES since fuel needs no recorded trajectory.
# Named to match nodrift_full_test.py's PLOT_FUEL_BAR, which draws the same
# chart. Not to be confused with evaluation_utilities.plot_fuel_comparison(),
# which is a box plot.
# True: save it.  
# False: skip it.
PLOT_FUEL_BAR = True

# Fuel chart options:

# How losing episodes are shown. Ignored when FUEL_TOP_N is set, which already filters to wins.
# True:  drop them, so failures do not skew the successful-dock fuel cost.
# False: show every episode, losses hatched.
FUEL_EXCLUDE_LOSSES = True

# How many episodes to show, picked from episodes both agents won, ranked by lowest combined fuel.
# An int: show that many.
# None:   show every episode, using FUEL_EXCLUDE_LOSSES instead.
FUEL_TOP_N = 10

# What the mean line and legend number cover.
# True:  every winning episode the agent ran, so the reported mean stays
#        representative even when FUEL_TOP_N narrows the chart.
# False: only the bars displayed.
FUEL_MEAN_OVER_ALL_EPISODES = True

# Figure style:

# Blue (Drift) / amber-orange (Nodrift), shared across all figures so agent identity stays consistent.
# Note: blue means "win" in drift_full_test.py's diagnostics gradient, a different figure.
AGENT_COLORS = {"A": "#0072B2", "B": "#E8A33D"}

# Fallback loss colors for the 3D start dots and fuel bars.
# One per agent so a loss never matches that agent's own win color.
FAIL_COLORS = {"A": "#FF3B30", "B": "#595959"}

# Per-agent linestyle for diagnostics and the fuel comparison's mean line
# (Drift solid, Nodrift dashed). The 3D plot keeps its own solid/dashed
# convention, which encodes win/loss instead.
AGENT_LINESTYLES = {"A": "-", "B": "--"}

# Marker shape for each agent's start-of-episode dot, so the two are
# distinguishable by shape as well as color where they land close together.
AGENT_MARKERS = {"A": "o", "B": "s"}

# Meters to nudge Nodrift's start dot along a diagonal so it does not render
# on top of Drift's (both share the same start state). Offsets the marker only. 0 disables.
START_MARKER_JITTER = 3.0

# (steps drawn, steps skipped) for Nodrift's 3D line under COLOR_TRAJECTORY_BY_THRUST.
# Dash linestyles do not render on that mode's one-step segments, so this drops segments instead.
# None or (0, 0) gives a solid line.
NODRIFT_DASH_PATTERN = (4, 3)

# Keep only every Nth burst triangle for Nodrift, whose early near-saturated thrust
# would otherwise cluster into a dense region. Drift's bursts are already sparse. 1 disables.
NODRIFT_TRIANGLE_STRIDE = 3

# Color each agent's start dot with the per-episode win/loss gradient used
# in plot_diagnostics, ranked within that agent's own group.
# True:  per-episode gradient, so individual episodes are distinguishable.
# False: flat agent_color() per dot.
TRAJECTORY_GRADIENT_BY_EPISODE = True

# Shade each diagnostics line a distinct tint (blue Drift win, orange
# Nodrift win, red loss), moving agent identity to linestyle.
# True:  per-episode gradient.
# False: one flat color per agent.
DIAGNOSTICS_GRADIENT_BY_EPISODE = True

# Color each trajectory by thrust magnitude (bright = hard thrust, dark = coast).
# Matches the same flag in drift_full_test.py and nodrift_full_test.py.
# True:  color by thrust.
# False: flat per-agent color.
COLOR_TRAJECTORY_BY_THRUST = True

# Scale line width by thrust and mark any step above THRUST_BURST_FRACTION
# of the episode's max, so brief hard thrusts stand out.
# True:  show burst markers.
# False: plain lines.
THRUST_BURST_MARKERS = True

# How much those burst markers pop. Both modes color by the thrust colormap.
# True:  enlarged, fully opaque, bold edge.
# False: smaller and subtler, blending into the line.
BRIGHT_THRUST_HIGHLIGHTS = True

# Fraction of max thrust above which a step gets a burst marker.
THRUST_BURST_FRACTION = 0.2

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

# COAST_THRUST_THRESHOLD, TRUE_COAST_THRESHOLD, ACTION_SATURATION_THRESHOLD,
# and APPROACH_CHECK_DISTANCE come from evaluation_utilities, not set here,
# so the comparison table's numbers stay consistent across scripts. Figure
# style stays per-script (this one's THRUST_BURST_FRACTION is deliberately
# lower than the others').

# Coast lookahead resolution, finer/longer by default than what the drift
# agent trained with.
# "10s_50step":  matches the drift agent's training setup (stage 8). Default.
# "1s_1000step": the original get_curriculum(9) setting. Useful for ablations.
DRIFT_EVAL_LOOKAHEAD = "10s_50step"

# The overrides DRIFT_EVAL_LOOKAHEAD selects between.
_DRIFT_LOOKAHEAD_OVERRIDES = {
    "1s_1000step": {},  # use get_curriculum(9)'s default values
    "10s_50step": {"drift_step_len": 10, "max_lookahead_len": 50},
}


# --- Execution ---
# Helper functions and the run itself. Configure using the settings above.

def resolve_drift_test_config(verbose=False) -> dict:
    """Build the drift test environment configuration with consistent lookahead settings.

    The script uses one shared lookahead configuration for start states and episodes,
    so all parts of the test use the same settings.

    Args:
        verbose: If True, prints the configuration before applying DRIFT_EVAL_LOOKAHEAD
            overrides. Usually kept False since the main startup already prints the
            final drift_step_len and max_lookahead_len values.

    Returns:
        A dictionary containing the environment configuration, with drift_step_len
        and max_lookahead_len set according to DRIFT_EVAL_LOOKAHEAD.
    """
    config, threshold = resolve_drift_eval_config(
        overrides=_DRIFT_LOOKAHEAD_OVERRIDES[DRIFT_EVAL_LOOKAHEAD], verbose=verbose)
    return config

def agent_color(tag: str, is_win: bool) -> str:
    """Resolve the flat color for one agent's episode, win or loss.

    Every agent uses its AGENT_COLORS entry on a win and its FAIL_COLORS
    entry on a loss, so losses don't blend into the agent's win color.

    Args:
        tag: "A" (Drift) or "B" (Nodrift).
        is_win: Whether this particular episode was a win.

    Returns:
        A hex color string.
    """
    return AGENT_COLORS[tag] if is_win else FAIL_COLORS[tag]


def build_gradient_cmaps() -> dict:
    """Build the shared per-agent, per-outcome win/loss gradient colormaps.

    Creates separate win and loss color gradients for each agent.
    Both plots use these gradients so colors stay consistent.

    Returns:
        {"A": {"win": cmap, "loss": cmap}, "B": {"win": cmap, "loss": cmap}}
    """
    from matplotlib.colors import LinearSegmentedColormap
    return {
        "A": {
            # Drift win: light blue -> navy. 
            # Drift loss: bright red -> maroon. 
            # Both stay within their own color family, distinct from Nodrift's amber win color.
            "win": LinearSegmentedColormap.from_list(
                "drift_win_blue",
                [(0.0, "#7EC8E3"), (0.4, "#2E86C1"), (0.7, "#005AB5"), (1.0, "#00246B")],
            ),
            "loss": LinearSegmentedColormap.from_list(
                "drift_loss_red", ["#FF6B57", "#FF3B30", "#D32F0A", "#7A1300"]
            ),
        },
        "B": {
            # Nodrift win: amber -> orange -> brown. 
            # Nodrift loss: light grey -> near-black, distinct from both Drift's 
            # red and its own orange win, so a Nodrift loss never reads as a win.
            "win": LinearSegmentedColormap.from_list(
                "nodrift_win_amber_orange",
                [(0.0, "#FFC966"), (0.55, "#E8A33D"), (0.8, "#C77800"), (1.0, "#8C5000")],
            ),
            "loss": LinearSegmentedColormap.from_list(
                "nodrift_loss_grey", ["#D9D9D9", "#9E9E9E", "#616161", "#262626"]
            ),
        },
    }


def evaluate_drift(model_paths: list, thresholds: list, label: str, seed: int,
                    start_states: list, stage_obs_time_norms: list = None) -> dict:
    """Run NUM_EPISODES paired episodes chaining the drift model curriculum.

    Each episode starts with the hardest model and moves to easier models
    after each successful stage. The final result is recorded for comparison
    with the nodrift agent.

    Args:
        model_paths: Chain of checkpoint paths, hardest (index 0) to
            easiest (last index).
        thresholds: (pos_thresh, speed_thresh) per model, same order.
        label: Short display name for this agent in printed output.
        seed: Base seed for the paired episodes (episode i uses seed + i).
        start_states: One shared start state per episode (see
            evaluation_utilities.resolve_start_states), used instead of
            this environment's own
            random sampling so both agents genuinely start from the
            same position and velocity each episode.
        stage_obs_time_norms: Per USE_PER_STAGE_OBS_TIME_NORM, same order
            as model_paths. None leaves the eval env's own obs[6] divisor
            untouched.

    Returns:
        A dict of aggregate results plus raw trajectories/outcomes for
        the first NUM_PLOT_EPISODES episodes, for plotting.
    """
    models = [PPO.load(str(p), device="cpu") for p in model_paths]
    # Match the eval env to what these checkpoints trained with. Obs-extending flags
    # come from observation size; reward-side flags come from run_metadata.json
    # (empty for older checkpoints). space_controls_obs is read first since it shifts the size mapping.
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
    # Print the override once because it applies to every episode.
    if DRIFT_DOCKING_POSITION_OVERRIDE is not None or DRIFT_DOCKING_SPEED_OVERRIDE is not None:
        _trained_pos = apply_dock_radius_override(
            thresholds[-1][0], space_controls_flags, "drift")
        override_dock_threshold(
            _trained_pos, thresholds[-1][1],
            DRIFT_DOCKING_POSITION_OVERRIDE, DRIFT_DOCKING_SPEED_OVERRIDE, label=label,
        )

    wins = 0
    fuels, fuels_l1, timesteps, episode_rewards = [], [], [], []
    elapsed_seconds_list = []  # real simulated time across the whole chain, per episode
    episode_rewards_weighted = []  # see the comment where this is appended
    failure_counts = {
        "crash": 0, "unsafe": 0, "out_of_bounds": 0, "fuel": 0,
        "timeout": 0, "unknown": 0,
    }
    trajectories, thrusts, speeds, outcomes = [], [], [], []
    all_outcomes = []  # win/loss for every episode, not just the recorded subset
    coast_steps = 0
    true_coast_steps = 0
    saturated_steps = 0
    violation_steps = 0  # steps exceeding env.env.velocity_limit(), DRL for Space Controls (2024)'s corresponding docking metric
    # Per-episode violation rate, averaged unweighted across episodes.
    # Unlike the step-weighted violation_fraction above, gives every episode equal say regardless of length.
    episode_violation_fractions = []
    # Which candidate the safety check picked when USE_SAFE_ACTION is on, showing if the shield is active.
    safe_action_counts = new_safe_action_counts()
    total_steps = 0
    entry_directions = []
    final_speeds = []  # speed at episode end, DRL for Space Controls (2024)'s other docking-specific metric

    for ep in range(NUM_EPISODES):
        curriculum = resolve_drift_test_config()
        env = DriftTestEnv(**drift_env_flags, **curriculum)
        # Override the curriculum's own start-state sampling with the
        # shared state for this episode, so both agents start identically.
        env.env.fixed_start = True
        env.env.fixed_state = start_states[ep].copy()
        obs, _ = env.reset(seed=seed + ep)

        ep_result = rollout_drift_chain_episode(
            env, obs, models, thresholds, model_paths,
            max_steps=MAX_STEPS_PER_EPISODE,
            use_safe_action=USE_SAFE_ACTION,
            safe_action_counts=safe_action_counts,
            episode_idx=ep,
            use_action_noise=USE_ACTION_NOISE,
            print_stage_results=PRINT_STAGE_RESULTS,
            stage_obs_time_norms=stage_obs_time_norms,
            final_pos_override=DRIFT_DOCKING_POSITION_OVERRIDE,
            final_speed_override=DRIFT_DOCKING_SPEED_OVERRIDE,
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
        # A drift chain earns one dock payout and time-penalty sum per stage, nodrift earns just one.
        # Dividing by chain length is a rough normalization to the same per-episode scale.
        episode_rewards_weighted.append(ep_result["episode_reward"] / len(models))
        fuels.append(ep_result["fuel"])
        fuels_l1.append(ep_result["fuel_l1"])
        all_outcomes.append(outcomes_this_episode)
        final_speeds.append(ep_result["speeds"][-1])
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
        "fuels": fuels,        # per-episode Delta-v, vector norm (L2, m/s), secondary; all NUM_EPISODES
        "fuels_l1": fuels_l1,  # per-episode Delta-v, per-axis sum (L1, m/s), primary; all NUM_EPISODES
        "all_outcomes": all_outcomes,  # win/loss for every episode, all NUM_EPISODES
        # Per-episode step count and reward, all NUM_EPISODES, kept raw for export_episode_results_csv.
        "episode_lengths": timesteps,
        "episode_rewards": episode_rewards,
        "episode_rewards_weighted": episode_rewards_weighted,
        "mean_timesteps": float(np.mean(timesteps)),
        # Report simulated seconds. Coast steps can cover more time than
        # one step, so this differs from the episode-step count.
        "elapsed_seconds": elapsed_seconds_list,
        "mean_time_to_dock": float(np.mean(elapsed_seconds_list)),
        "median_time_to_dock": float(np.median(elapsed_seconds_list)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "median_episode_reward": float(np.median(episode_rewards)),
        "mean_episode_reward_weighted": float(np.mean(episode_rewards_weighted)),
        "median_episode_reward_weighted": float(np.median(episode_rewards_weighted)),
        "coast_fraction": coast_steps / total_steps if total_steps else 0.0,
        "true_coast_fraction": true_coast_steps / total_steps if total_steps else 0.0,
        "saturation_fraction": saturated_steps / total_steps if total_steps else 0.0,
        # DRL for Space Controls (2024) (arXiv 2405.12355) reports these two docking metrics directly
        # (Table II: "Violation (%)" and "Final Speed (m/s)"), tracked here for comparison.
        "violation_fraction": violation_steps / total_steps if total_steps else 0.0,
        # Same metric, unweighted by episode length: mean of each episode's own violation rate,
        # instead of one rate pooled over every step.
        "violation_fraction_episode_mean": (
            float(np.mean(episode_violation_fractions))
            if episode_violation_fractions else 0.0),
        **summarize_safe_action_counts(safe_action_counts),
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


def evaluate_nodrift(model_path: str, label: str, seed: int, start_states: list) -> dict:
    """Run NUM_EPISODES paired episodes for the nodrift checkpoint.

    Mirrors nodrift_full_test.py's evaluate_checkpoint(), adapted to this
    script's own config constants and to label wins "win" (not
    "direct_dock") so both agents' outcomes compare directly in the
    shared plotting functions below.

    Args:
        model_path: Full path to the nodrift checkpoint file.
        label: Short display name for this agent in printed output.
        seed: Base seed for the paired episodes (episode i uses seed + i).
        start_states: One shared start state per episode (see
            evaluation_utilities.resolve_start_states), used instead of
            this environment's own
            random sampling so both agents genuinely start from the
            same position and velocity each episode.

    Returns:
        A dict of aggregate results plus raw trajectories/outcomes for
        the first NUM_PLOT_EPISODES episodes, for plotting.
    """
    model = PPO.load(model_path, device="cpu")
    expected_obs_size = model.observation_space.shape[0]
    # space_controls_obs must be read BEFORE resolve_obs_flags: it subtracts an
    # observation element, so size alone does not identify the flags.
    space_controls_flags = resolve_env_flags_from_metadata(
        model_path, keys=SPACE_CONTROLS_FLAG_KEYS)
    saferl_obs, custom_braking_margin_obs = resolve_obs_flags(
        expected_obs_size, space_controls_obs=space_controls_flags.get("space_controls_obs", False)
    )
    # Reward-shape flags don't change observation size, so they're read from metadata explicitly,
    # or the checkpoint runs under the wrong reward with no error.
    reward_flags = resolve_env_flags_from_metadata(model_path, keys=NODRIFT_REWARD_FLAG_KEYS)
    config = resolve_nodrift_env_config(model_path, FALLBACK_ENV_CONFIG)
    env = SpaceCraftDockingEnv3D(
        saferl_obs=saferl_obs,
        custom_braking_margin_obs=custom_braking_margin_obs,
        **space_controls_flags,
        **reward_flags,
        **config,
    )
    # Apply overrides after creating the environment. The constructor replaces
    # pos_thresh when space_controls_dock_radius is enabled, so changing the
    # config dictionary first would have no effect.
    env.dock_dist, env.dock_speed = override_dock_threshold(
        env.dock_dist, env.dock_speed,
        NODRIFT_DOCKING_POSITION_OVERRIDE, NODRIFT_DOCKING_SPEED_OVERRIDE, label=label,
    )

    wins = 0
    fuels, fuels_l1, timesteps, episode_rewards = [], [], [], []
    # Nodrift never chains, so there's nothing to weight against. Kept identical to episode_rewards
    # so the shared table row has a matching key on both sides.
    episode_rewards_weighted = episode_rewards
    failure_counts = {
        "crash": 0, "unsafe": 0, "out_of_bounds": 0, "fuel": 0,
        "timeout": 0, "unknown": 0,
    }
    trajectories, thrusts, speeds, outcomes = [], [], [], []
    all_outcomes = []  # win/failure-reason for every episode, not just the recorded subset
    coast_steps = 0
    true_coast_steps = 0
    saturated_steps = 0
    violation_steps = 0  # steps exceeding env.velocity_limit(), DRL for Space Controls (2024)'s corresponding docking metric
    # Per-episode violation rate, averaged unweighted across episodes.
    # Unlike the step-weighted violation_fraction above, this reflects each episode individually.
    episode_violation_fractions = []
    total_steps = 0
    entry_directions = []
    final_speeds = []  # speed at episode end, DRL for Space Controls (2024)'s other docking-specific metric

    for ep in range(NUM_EPISODES):
        # Override this env's start-state sampling with the shared state so both agents start identically.
        env.fixed_start = True
        env.fixed_state = start_states[ep].copy()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        episode_violation_steps = 0
        episode_reward = 0.0
        record = ep < NUM_PLOT_EPISODES
        trajectory = [env.state[0:3].copy()] if ((PLOT_TRAJECTORIES or PLOT_DIAGNOSTICS) and record) else None
        thrust_trace = [0.0] if trajectory is not None else None
        speed_trace = [float(np.linalg.norm(env.state[3:6]))] if trajectory is not None else None
        entry_recorded = False

        while not done:
            action, _ = model.predict(obs[:expected_obs_size], deterministic=True)
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, term, trunc, _ = env.step(action)
            steps += 1
            episode_reward += reward
            done = term or trunc
            if steps >= MAX_STEPS_PER_EPISODE:
                break
            thrust_mag = float(np.linalg.norm(action))
            total_steps += 1
            if thrust_mag < COAST_THRUST_THRESHOLD:
                coast_steps += 1
            if thrust_mag < TRUE_COAST_THRESHOLD:
                true_coast_steps += 1
            if np.max(np.abs(action)) > ACTION_SATURATION_THRESHOLD:
                saturated_steps += 1

            position = env.state[0:3]
            distance = float(np.linalg.norm(position))
            current_speed = float(np.linalg.norm(env.state[3:6]))
            if current_speed > env.velocity_limit(distance):
                violation_steps += 1
                episode_violation_steps += 1

            if not entry_recorded:
                if distance <= APPROACH_CHECK_DISTANCE and distance > 0:
                    entry_directions.append(position.copy() / distance)
                    entry_recorded = True

            if trajectory is not None:
                trajectory.append(env.state[0:3].copy())
                thrust_trace.append(thrust_mag)
                speed_trace.append(current_speed)

        timesteps.append(steps)
        fuels.append(env.fuel_used)
        fuels_l1.append(env.fuel_used_l1)
        episode_rewards.append(episode_reward)
        final_speeds.append(float(np.linalg.norm(env.state[3:6])))
        episode_violation_fractions.append(
            episode_violation_steps / steps if steps else 0.0)

        if env.is_docked():
            wins += 1
            outcome = "win"
        else:
            outcome = classify_failure_nodrift(env)
            failure_counts[outcome] += 1
        all_outcomes.append(outcome)

        if trajectory is not None:
            trajectories.append(trajectory)
            thrusts.append(thrust_trace)
            speeds.append(speed_trace)
            outcomes.append(outcome)

    print(f"  Wins: {wins}/{NUM_EPISODES} ({100 * wins / NUM_EPISODES:.1f}%)")

    return {
        "label": label,
        # Display-ready name, matching the drift side (which stores a
        # describe_drift_chain() string) so both print the same way.
        "checkpoint": os.path.basename(model_path),
        "wins": wins,
        "total": NUM_EPISODES,
        "win_rate": wins / NUM_EPISODES,
        **fuel_statistics(fuels, fuels_l1),
        "fuels": fuels,        # per-episode Delta-v, vector norm (L2, m/s), secondary; all NUM_EPISODES
        "fuels_l1": fuels_l1,  # per-episode Delta-v, per-axis sum (L1, m/s), primary; all NUM_EPISODES
        "all_outcomes": all_outcomes,  # win/loss for every episode, all NUM_EPISODES
        # Per-episode step count and reward, all NUM_EPISODES, kept raw for export_episode_results_csv.
        "episode_lengths": timesteps,
        "episode_rewards": episode_rewards,
        "episode_rewards_weighted": episode_rewards_weighted,
        "mean_timesteps": float(np.mean(timesteps)),
        # step_len is 1 for every nodrift episode here, so this equals
        # mean_timesteps. Uses the drift side's key name so the shared
        # comparison table reads both the same way.
        "elapsed_seconds": [t * env.step_len for t in timesteps],
        "mean_time_to_dock": float(np.mean(timesteps)) * env.step_len,
        "median_time_to_dock": float(np.median(timesteps)) * env.step_len,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "median_episode_reward": float(np.median(episode_rewards)),
        "mean_episode_reward_weighted": float(np.mean(episode_rewards_weighted)),
        "median_episode_reward_weighted": float(np.median(episode_rewards_weighted)),
        "coast_fraction": coast_steps / total_steps if total_steps else 0.0,
        "true_coast_fraction": true_coast_steps / total_steps if total_steps else 0.0,
        "saturation_fraction": saturated_steps / total_steps if total_steps else 0.0,
        # DRL for Space Controls (2024) (arXiv 2405.12355) reports these two docking metrics directly
        # (Table II: "Violation (%)" and "Final Speed (m/s)"), tracked here for comparison.
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


def plot_comparison(result_a: dict, result_b: dict, seed: int, show: bool = True) -> None:
    """Save a 3D trajectory plot with both agents' episodes overlaid.

    Deliberately local, not a call to
    evaluation_utilities.plot_paired_trajectories_3d(): this script needs
    per-episode gradient coloring, agent-specific markers, a start-marker
    jitter, and a dash-pattern technique the shared function doesn't support.
    Check whether a rendering fix here also belongs in the shared version.

    Agent identity is carried by line style: Drift solid, Nodrift dashed
    (gapped segments per NODRIFT_DASH_PATTERN when
    COLOR_TRAJECTORY_BY_THRUST is True, since a real dashed linestyle
    doesn't render reliably on that mode's Line3DCollection). Line color
    is thrust magnitude or outcome, depending on COLOR_TRAJECTORY_BY_THRUST.
    Start-dot color and shape carry agent identity and, with
    TRAJECTORY_GRADIENT_BY_EPISODE on, the same per-episode win/loss
    gradient as plot_diagnostics, so episodes stay distinguishable even
    where both agents' start dots land near the same point.

    Args:
        result_a: Return value of evaluate_drift().
        result_b: Return value of evaluate_nodrift().
        seed: The resolved seed the paired episodes ran with.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    fig = plt.figure(figsize=(13, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    labels_used = set()
    burst_labeled = False
    thrust_line = None

    if TRAJECTORY_GRADIENT_BY_EPISODE:
        gradient_cmaps = build_gradient_cmaps()
        # Ranked within each agent's own win/loss group, matching
        # plot_diagnostics's convention: the 2nd win gets the 2nd shade.
        gradient_ranks = {}
        for tag, result in (("A", result_a), ("B", result_b)):
            gradient_ranks[tag] = {
                "n_wins": sum(1 for o in result["outcomes"] if o == "win"),
                "n_losses": sum(1 for o in result["outcomes"] if o != "win"),
                "win_rank": 0,
                "loss_rank": 0,
            }

    if COLOR_TRAJECTORY_BY_THRUST:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        # Build one color scale from the hardest thrust across BOTH agents,
        # so the same shade means the same thrust on every line here.
        max_thrust = max_thrust_across_results(result_a, result_b)
        thrust_cmap, thrust_norm = build_thrust_colormap(THRUST_COLORMAP, max_thrust)

    for tag, result in (("A", result_a), ("B", result_b)):
        for traj, thrust, outcome in zip(result["trajectories"], result["thrusts"], result["outcomes"]):
            traj = np.array(traj)
            if len(traj) < 2:
                continue
            is_win = outcome == "win"
            key = (tag, is_win)
            label = None
            if key not in labels_used:
                label = f"{result['label']} ({'win' if is_win else 'loss'})"
                labels_used.add(key)

            if TRAJECTORY_GRADIENT_BY_EPISODE:
                ranks = gradient_ranks[tag]
                if is_win:
                    start_color = gradient_cmaps[tag]["win"](ranks["win_rank"] / max(ranks["n_wins"] - 1, 1))
                    ranks["win_rank"] += 1
                else:
                    start_color = gradient_cmaps[tag]["loss"](ranks["loss_rank"] / max(ranks["n_losses"] - 1, 1))
                    ranks["loss_rank"] += 1
            else:
                start_color = agent_color(tag, is_win)

            # Nodrift's start dot is nudged along a diagonal so it doesn't land on Drift's (same start state).
            # Only the marker moves, the line still starts from the real position.
            start_point = traj[0].copy()
            if tag == "B" and START_MARKER_JITTER:
                start_point = start_point + START_MARKER_JITTER / np.sqrt(3)

            if COLOR_TRAJECTORY_BY_THRUST:
                points = traj.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_thrust = np.array(thrust[1:len(traj)])
                # A dashed linestyle doesn't work here: matplotlib applies the dash pattern per-path,
                # and each path is one tiny segment, so most render fully "on" with no visible gaps.
                # Drop a periodic chunk of Nodrift's segments instead to cut real gaps.
                burst_source_points = traj[1:]  # kept in lockstep with segments/segment_thrust below
                if tag == "B" and NODRIFT_DASH_PATTERN and NODRIFT_DASH_PATTERN[1] > 0:
                    dash_on, dash_off = NODRIFT_DASH_PATTERN
                    period = dash_on + dash_off
                    keep = (np.arange(len(segments)) % period) < dash_on
                    segments = segments[keep]
                    segment_thrust = segment_thrust[keep]
                    burst_source_points = burst_source_points[keep]
                lc = Line3DCollection(segments, cmap=thrust_cmap, norm=thrust_norm, alpha=0.9)
                lc.set_array(segment_thrust)
                if THRUST_BURST_MARKERS:
                    lc.set_linewidth(1.5 + 4.5 * (segment_thrust / max_thrust))
                else:
                    lc.set_linewidth(2.0)
                ax.add_collection3d(lc)
                thrust_line = lc

                if THRUST_BURST_MARKERS:
                    burst_mask = segment_thrust >= THRUST_BURST_FRACTION * max_thrust
                    if np.any(burst_mask):
                        burst_points = burst_source_points[burst_mask]
                        burst_values = segment_thrust[burst_mask]
                        if tag == "B" and NODRIFT_TRIANGLE_STRIDE > 1:
                            burst_points = burst_points[::NODRIFT_TRIANGLE_STRIDE]
                            burst_values = burst_values[::NODRIFT_TRIANGLE_STRIDE]
                        burst_label = None
                        if not burst_labeled:
                            burst_label = "Hard thrust"
                            burst_labeled = True
                        # Always colored by thrust magnitude, same as the line.
                        # BRIGHT_THRUST_HIGHLIGHTS only changes how much the markers pop (size/edge/alpha).
                        if BRIGHT_THRUST_HIGHLIGHTS:
                            ax.scatter(burst_points[:, 0], burst_points[:, 1], burst_points[:, 2],
                                       c=burst_values, cmap=thrust_cmap, norm=thrust_norm,
                                       marker="^", s=90, edgecolors="black", linewidths=0.9,
                                       alpha=1.0, zorder=6, label=burst_label)
                        else:
                            ax.scatter(burst_points[:, 0], burst_points[:, 1], burst_points[:, 2],
                                       c=burst_values, cmap=thrust_cmap, norm=thrust_norm,
                                       marker="^", s=55, edgecolors="black", linewidths=0.5,
                                       alpha=0.95, zorder=6, label=burst_label)

                ax.scatter(*start_point, color=start_color, marker=AGENT_MARKERS[tag], s=35,
                           alpha=0.9, edgecolors="black", linewidths=0.5, label=label, zorder=7)
            else:
                linestyle = "-" if tag == "A" else "--"
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=agent_color(tag, is_win),
                         linestyle=linestyle, linewidth=1.2, alpha=0.6, label=label)
                ax.scatter(*start_point, color=start_color, marker=AGENT_MARKERS[tag], s=30,
                           alpha=0.9, edgecolors="black", linewidths=0.5, zorder=7)

    if thrust_line is not None:
        if THRUST_COLORBAR_AT_BOTTOM:
            # orientation is set explicitly even though location="bottom" implies it.
            # shrink is ~30% smaller than the vertical bar's, pad raised to clear the 3D axes' tick labels.
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
    ax.set_title(
        f"Drift vs. Nodrift Comparison (seed {seed})\n"
        f"A ({result_a['label']}): {result_a['checkpoint']}  "
        f"({result_a['wins']}/{result_a['total']})\n"
        f"B ({result_b['label']}): {result_b['checkpoint']}  "
        f"({result_b['wins']}/{result_b['total']})",
        fontsize=10,
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    # tight_layout() alone can still let a tall 3D box's title clip off
    # the top of the window; reserve explicit headroom for it.
    fig.subplots_adjust(top=0.92)

    save_and_show_figure(BASE_DIR, "drift_vs_nodrift_test",
                         "drift_vs_nodrift_comparison", "trajectory plot", show=show)


def plot_diagnostics(result_a: dict, result_b: dict, seed: int, show: bool = True) -> None:
    """Save a 2x2 per-step diagnostics figure with both agents overlaid.

    Deliberately local, not a call to
    evaluation_utilities.plot_paired_diagnostics() (used by the other two
    paired-comparison scripts): this script needs per-episode gradient
    coloring and agent-specific linestyle the shared function doesn't
    support. If you fix a rendering bug or add a feature to the shared
    version, check whether this copy needs the same change.

    Panels: thrust magnitude vs. time, thrust magnitude vs. distance to the
    chief, distance vs. time, and speed vs. time, for the first
    DIAG_NUM_EPISODES recorded episodes per agent. Same layout as
    nodrift_full_test.py's plot_diagnostics().

    Args:
        result_a: Return value of evaluate_drift().
        result_b: Return value of evaluate_nodrift().
        seed: The resolved seed the paired episodes ran with.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_thrust_time, ax_thrust_dist), (ax_dist_time, ax_speed_time) = axes

    # Agent identity is carried by linestyle so it stays visible even when
    # color is spent on the win/loss gradient below.
    linestyles = AGENT_LINESTYLES

    if DIAGNOSTICS_GRADIENT_BY_EPISODE:
        gradient_cmaps = build_gradient_cmaps()

    for tag, result in (("A", result_a), ("B", result_b)):
        n_episodes = min(DIAG_NUM_EPISODES, len(result["trajectories"]))
        if DIAGNOSTICS_GRADIENT_BY_EPISODE:
            win_cmap = gradient_cmaps[tag]["win"]
            loss_cmap = gradient_cmaps[tag]["loss"]
            # Ranked within this agent's own win/loss groups, so the 2nd win gets the 2nd shade.
            n_wins = sum(1 for i in range(n_episodes) if result["outcomes"][i] == "win")
            n_losses = n_episodes - n_wins
            win_rank = loss_rank = 0

        for i in range(n_episodes):
            traj = np.array(result["trajectories"][i])
            thrust = np.array(result["thrusts"][i])
            speed = np.array(result["speeds"][i])
            if len(traj) < 2:
                continue
            distance = np.linalg.norm(traj, axis=1)
            time = np.arange(len(traj))

            is_win = result["outcomes"][i] == "win"
            if DIAGNOSTICS_GRADIENT_BY_EPISODE:
                if is_win:
                    color = win_cmap(win_rank / max(n_wins - 1, 1))
                    win_rank += 1
                else:
                    color = loss_cmap(loss_rank / max(n_losses - 1, 1))
                    loss_rank += 1
            else:
                color = agent_color(tag, is_win)

            # Legend handles are built separately below (as gradient swatches when
            # DIAGNOSTICS_GRADIENT_BY_EPISODE is on), so individual lines don't need their own label.
            ax_thrust_time.plot(time, thrust, color=color, alpha=0.6,
                                linewidth=1.2, linestyle=linestyles[tag])
            ax_thrust_dist.plot(distance, thrust, color=color, alpha=0.6,
                                linewidth=1.2, linestyle=linestyles[tag])
            ax_dist_time.plot(time, distance, color=color, alpha=0.6,
                              linewidth=1.2, linestyle=linestyles[tag])
            ax_speed_time.plot(time, speed, color=color, alpha=0.6,
                               linewidth=1.2, linestyle=linestyles[tag])

    for ax in (ax_thrust_time, ax_thrust_dist):
        ax.axhline(COAST_THRUST_THRESHOLD, color="black", linestyle=":",
                   linewidth=1.0, alpha=0.7)

    ax_thrust_time.set_xlabel("Time (steps = seconds)")
    ax_thrust_time.set_ylabel("Thrust magnitude (Newtons, L2 norm)")
    ax_thrust_time.set_title("Thrust vs. time")

    ax_thrust_dist.set_xlabel("Distance to chief (m)")
    ax_thrust_dist.set_ylabel("Thrust magnitude (Newtons, L2 norm)")
    ax_thrust_dist.set_title("Thrust vs. distance")
    ax_thrust_dist.invert_xaxis()

    ax_dist_time.set_xlabel("Time (steps = seconds)")
    ax_dist_time.set_ylabel("Distance to chief (m)")
    ax_dist_time.set_title("Distance vs. time")

    ax_speed_time.set_xlabel("Time (steps = seconds)")
    ax_speed_time.set_ylabel("Speed (m/s)")
    ax_speed_time.set_title("Speed vs. time")

    from matplotlib.lines import Line2D
    if DIAGNOSTICS_GRADIENT_BY_EPISODE:
        # A true gradient line isn't renderable as a single legend swatch, so use each cmap's
        # color partway along its range (not the washed-out lightest stop) as a representative color.
        legend_handles = [
            Line2D([0], [0], color=gradient_cmaps["A"]["win"](0.65), linestyle=linestyles["A"], linewidth=2),
            Line2D([0], [0], color=gradient_cmaps["A"]["loss"](0.55), linestyle=linestyles["A"], linewidth=2),
            Line2D([0], [0], color=gradient_cmaps["B"]["win"](0.55), linestyle=linestyles["B"], linewidth=2),
            Line2D([0], [0], color=gradient_cmaps["B"]["loss"](0.55), linestyle=linestyles["B"], linewidth=2),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], color=agent_color("A", True), linestyle=linestyles["A"], linewidth=2),
            Line2D([0], [0], color=agent_color("A", False), linestyle=linestyles["A"], linewidth=2),
            Line2D([0], [0], color=agent_color("B", True), linestyle=linestyles["B"], linewidth=2),
            Line2D([0], [0], color=agent_color("B", False), linestyle=linestyles["B"], linewidth=2),
        ]
    legend_labels = [
        f"{result_a['label']} (win)", f"{result_a['label']} (loss)",
        f"{result_b['label']} (win)", f"{result_b['label']} (loss)",
    ]
    for ax in (ax_thrust_time, ax_thrust_dist, ax_dist_time, ax_speed_time):
        ax.legend(handles=legend_handles, labels=legend_labels, loc="upper right", fontsize=7)

    linestyle_note = (
        f"; solid = {result_a['label']}, dashed = {result_b['label']}"
        if DIAGNOSTICS_GRADIENT_BY_EPISODE else ""
    )
    fig.suptitle(
        f"Per-step diagnostics: {result_a['label']} vs {result_b['label']} (seed {seed})\n"
        f"First {DIAG_NUM_EPISODES} {plural_word(DIAG_NUM_EPISODES, 'episode')} each{linestyle_note}\n"
        f"Dotted line = coast threshold {COAST_THRUST_THRESHOLD} Newtons",
        fontsize=11,
    )
    plt.tight_layout()

    save_and_show_figure(BASE_DIR, "drift_vs_nodrift_test",
                         "drift_vs_nodrift_diagnostics", "diagnostics plot", show=show)


def plot_approach_directions(result_a: dict, result_b: dict, seed: int,
                             show: bool = True) -> None:
    """Save a scatter plot of each episode's approach direction.

    Thin wrapper over
    evaluation_utilities.plot_paired_approach_directions(), which owns the
    plot itself so all three paired-comparison scripts render it
    identically. Only this script's colors and output paths differ.

    Args:
        result_a: Result dict for side A.
        result_b: Result dict for side B.
        seed: The resolved seed the paired episodes ran with.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    plot_paired_approach_directions(
        result_a, result_b,
        colors=AGENT_COLORS,
        base_dir=BASE_DIR,
        subfolder="drift_vs_nodrift_test",
        filename_prefix="drift_vs_nodrift_approach_directions",
        check_distance=APPROACH_CHECK_DISTANCE,
        seed=seed,
        show=show,
    )


def plot_fuel_bar(result_a: dict, result_b: dict, seed: int, show: bool = True) -> None:
    """Save a grouped bar chart of total fuel used per episode, both agents.

    One bar pair per selected episode (both agents saw the same start
    state for a given episode index, so a side-by-side pair is a direct
    apples-to-apples comparison). Plots the L1 norm, proportional to the
    propellant the fixed per-axis thrusters use.

    When FUEL_TOP_N is set, the episodes shown are the FUEL_TOP_N with
    the lowest combined fuel (A + B) among episodes BOTH agents won,
    e.g. 15 episodes run but only the best 5 plotted, instead of
    FUEL_EXCLUDE_LOSSES's independent per-agent win/loss filtering over
    every episode.

    Args:
        result_a: Return value of evaluate_drift().
        result_b: Return value of evaluate_nodrift().
        seed: The resolved seed the paired episodes ran with.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    fig, ax = plt.subplots(figsize=(max(10, (FUEL_TOP_N or NUM_EPISODES) * 0.6), 6))

    n = len(result_a["fuels"])
    bar_width = 0.38
    # Uses all_outcomes (every episode), not outcomes (only the recorded-trajectory subset),
    # since fuel totals exist for every episode.
    outcomes_a = result_a["all_outcomes"]
    outcomes_b = result_b["all_outcomes"]

    if FUEL_TOP_N is not None:
        # Both agents must have won a given episode for a fair fuel comparison. Rank by combined
        # fuel (lowest = most efficient pair) and keep only the best FUEL_TOP_N.
        both_won = [i for i in range(n) if outcomes_a[i] == "win" and outcomes_b[i] == "win"]
        both_won.sort(key=lambda i: result_a["fuels_l1"][i] + result_b["fuels_l1"][i])
        shown = sorted(both_won[:FUEL_TOP_N])
    else:
        shown = list(range(n))
    # Bars sit at compact positions 0..len(shown)-1, not each episode's real index, so a
    # FUEL_TOP_N of 5 out of 15 renders as 5 tight pairs instead of scattered bars.
    # X tick labels still show the real episode number.
    positions = np.arange(len(shown))

    for offset, tag, result, outcomes in ((-1, "A", result_a, outcomes_a), (1, "B", result_b, outcomes_b)):
        shown_fuels = [result["fuels_l1"][i] for i in shown]
        if FUEL_MEAN_OVER_ALL_EPISODES:
            # Every winning episode this agent ran, not just the displayed bars, so the mean
            # doesn't quietly become "mean of the best N" once FUEL_TOP_N narrows the chart.
            mean_fuels = [result["fuels_l1"][i] for i in range(n) if outcomes[i] == "win"]
        else:
            mean_fuels = shown_fuels
        # No wins to average over (e.g. an undertrained or collapsed checkpoint) has no fuel mean.
        # Say so plainly instead of a "(mean nan)" legend entry.
        mean_label = "no wins" if not mean_fuels else None
        if mean_fuels:
            mean_shown = float(np.mean(mean_fuels))
            mean_label = f"mean {mean_shown:.3f}"
        # Drift's loss bars are colored red (agent_color); Nodrift keeps
        # its own color on a loss and relies on the hatch below instead.
        bar_colors = [agent_color(tag, outcomes[i] == "win") for i in shown]

        exclude_losses = FUEL_TOP_N is not None or FUEL_EXCLUDE_LOSSES
        bars = ax.bar(positions + offset * bar_width / 2, shown_fuels, width=bar_width,
                      color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5,
                      label=f"{result['label']} ({mean_label})")
        if not exclude_losses:
            # Hatch losing bars so a short bar that is really a loss does
            # not read as fuel-efficient.
            for bar, i in zip(bars, shown):
                if outcomes[i] != "win":
                    bar.set_hatch("//")

        if mean_fuels:
            ax.axhline(mean_shown, color=AGENT_COLORS[tag], linestyle=AGENT_LINESTYLES[tag],
                      linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Delta-v used, per-axis sum (L1, m/s)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in shown])
    if FUEL_TOP_N is not None:
        selection_note = f"top {len(shown)} of {n} by combined fuel, both agents won; "
    else:
        selection_note = "wins only; " if FUEL_EXCLUDE_LOSSES else "hatched bars = loss; "
    mean_note = f"mean over all {n} episodes" if FUEL_MEAN_OVER_ALL_EPISODES else "mean over episodes shown"
    ax.set_title(
        f"Total Delta-v (L1) per episode: {result_a['label']} vs {result_b['label']} (seed {seed})\n"
        f"({len(shown)} {plural_word(len(shown), 'episode')} shown; {selection_note.rstrip('; ')})\n"
        f"Horizontal lines = {mean_note}"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    save_and_show_figure(BASE_DIR, "drift_vs_nodrift_test",
                         "drift_vs_nodrift_fuel_comparison", "fuel comparison plot", show=show)


def print_comparison_table(result_a: dict, result_b: dict, seed: int) -> None:
    """Print the side-by-side comparison table for this script's two results.

    Thin wrapper over evaluation_utilities.print_paired_comparison_table(),
    which owns the row list, the auto-sized columns, and the fuel
    percentile breakdown so all three paired-comparison scripts stay
    identical in format. Rows whose metric a result does not carry are
    skipped there automatically.

    Args:
        result_a: Result dict for side A.
        result_b: Result dict for side B.
        seed: The resolved seed the paired episodes ran with.
    """
    print_paired_comparison_table(result_a, result_b, seed, NUM_EPISODES)


if __name__ == "__main__":
    drift_model_paths = list(reversed(
        resolve_checkpoint_paths(DRIFT_CHECKPOINT, str(CHECKPOINT_ROOT), num_models=DRIFT_NUM_MODELS,
                                 folder_prefix="safe_PPO")
    ))
    drift_model_paths = [Path(p) for p in drift_model_paths]
    drift_thresholds = [resolve_drift_stage_thresholds(str(p)) for p in drift_model_paths]
    drift_obs_time_norms = (
        [resolve_drift_stage_obs_time_norm(str(p)) for p in drift_model_paths]
        if USE_PER_STAGE_OBS_TIME_NORM else None)

    # folder_prefix keeps "latest" on standalone nodrift runs only. Use
    # "run:nodrift_curriculum_PPO_<N>" for a curriculum baseline. Two
    # prefixes because standalone folders use either nodrift_PPO_{run} or
    # nodrift_standalone_PPO_{run}; str.startswith() accepts a tuple.
    nodrift_model_path = resolve_checkpoint_paths(
        NODRIFT_CHECKPOINT, str(CHECKPOINT_ROOT), num_models=1,
        folder_prefix=("nodrift_PPO", "nodrift_standalone_PPO"))[-1]

    # folder_prefix above only constrains "latest"; an explicit "run:" spec or exact path bypasses
    # it, so either constant could still resolve to the wrong family. Check here for a clear error
    # instead of a confusing crash deep in env construction.
    drift_agent_mode = detect_agent_mode(str(drift_model_paths[0]))
    if drift_agent_mode != "drift":
        raise ValueError(
            f"DRIFT_CHECKPOINT ({DRIFT_CHECKPOINT!r}) resolved to a "
            f"checkpoint detected as {drift_agent_mode!r}: "
            f"{drift_model_paths[0]}. Point it at a drift run (folder "
            f"prefix safe_PPO)."
        )
    nodrift_agent_mode = detect_agent_mode(nodrift_model_path)
    if nodrift_agent_mode != "nodrift":
        raise ValueError(
            f"NODRIFT_CHECKPOINT ({NODRIFT_CHECKPOINT!r}) resolved to a "
            f"checkpoint detected as {nodrift_agent_mode!r}: "
            f"{nodrift_model_path}. Point it at a nodrift run (folder "
            f"prefix nodrift)."
        )

    # Name the nodrift training style from its run folder. re.fullmatch
    # anchors both ends, so a standalone pattern cannot match the longer
    # curriculum name. Unrecognized folders keep the plain label.
    nodrift_run_folder = Path(nodrift_model_path).parent.name
    label_a, label_b = LABEL_A, LABEL_B
    if SHOW_NODRIFT_TRAINING_STYLE:
        if re.fullmatch(r"nodrift_(PPO|standalone_PPO)_\d+", nodrift_run_folder):
            label_b = f"{LABEL_B} (standalone)"
        elif re.fullmatch(r"nodrift_curriculum_PPO_\d+", nodrift_run_folder):
            label_b = f"{LABEL_B} (curriculum)"

    resolved_seed = resolve_and_print_seed(SEED, leading_newline=True)

    # Resolved once here (rather than left to its first use further down)
    # so the startup printout can report which lookahead resolution the
    # coast check is running under this evaluation.
    drift_test_config = resolve_drift_test_config()

    print(f"{label_a}: chain of {len(drift_model_paths)} models, "
          f"hardest {drift_model_paths[0].name}")
    print(f"{label_b}: {Path(nodrift_model_path).name}")
    print(f"Drift coast lookahead: {DRIFT_EVAL_LOOKAHEAD} "
          f"(drift_step_len={drift_test_config['drift_step_len']}, "
          f"max_lookahead_len={drift_test_config['max_lookahead_len']})")
    print()
    print("Drift policy (chain order: farthest start first, tightest final dock last):")
    for path, (pos_thresh, speed_thresh) in zip(drift_model_paths, drift_thresholds):
        print(f"  stage {extract_stage(str(path)):<3} {path.name}  "
              f"(pos_thresh={pos_thresh}, speed_thresh={speed_thresh})")

    print()
    print(f"{plural_word(NUM_EPISODES, 'Episode')} per agent: {NUM_EPISODES}")

    # Both agents use the same start state in each episode.
    # When sampling starts, use bounds that fit both agents. TEST_SET overrides these bounds.
    nodrift_config = resolve_nodrift_env_config(nodrift_model_path, FALLBACK_ENV_CONFIG, verbose=False)

    # The final stage sets the chain's win condition.
    # Apply the dock-radius flag because it replaces pos_thresh in the environment.
    # The recorded configuration still contains the original pos_thresh value.
    _drift_final_pos, _drift_final_speed = drift_thresholds[-1]
    _drift_dock_radius_flags = resolve_env_flags_from_metadata(
        str(drift_model_paths[-1]), keys=("space_controls_dock_radius",))
    _nodrift_dock_radius_flags = resolve_env_flags_from_metadata(
        nodrift_model_path, keys=("space_controls_dock_radius",))
    # Apply any test overrides before checking the docking conditions.
    # The warning compares the settings used in this test.
    _drift_effective_pos, _drift_effective_speed = override_dock_threshold(
        apply_dock_radius_override(_drift_final_pos, _drift_dock_radius_flags, "drift"),
        _drift_final_speed, DRIFT_DOCKING_POSITION_OVERRIDE, DRIFT_DOCKING_SPEED_OVERRIDE, verbose=False,
    )
    _nodrift_effective_pos, _nodrift_effective_speed = override_dock_threshold(
        apply_dock_radius_override(nodrift_config["pos_thresh"], _nodrift_dock_radius_flags, "nodrift"),
        nodrift_config["speed_thresh"], NODRIFT_DOCKING_POSITION_OVERRIDE, NODRIFT_DOCKING_SPEED_OVERRIDE, verbose=False,
    )
    print()
    print("Current docking conditions:")
    print(f"  {label_a}: distance <= {_drift_effective_pos:g} m, "
          f"speed <= {_drift_effective_speed:g} m/s")
    print(f"  {label_b}: distance <= {_nodrift_effective_pos:g} m, "
          f"speed <= {_nodrift_effective_speed:g} m/s")
    warn_if_win_condition_differs(
        label_a, _drift_effective_pos, _drift_effective_speed,
        label_b, _nodrift_effective_pos, _nodrift_effective_speed,
    )

    shared_start_states, start_state_provenance = resolve_start_states(
        TEST_SET, NUM_EPISODES, resolved_seed,
        min_pos_bound=max(drift_test_config["min_init_pos_bound"],
                          nodrift_config["min_init_pos_bound"]),
        max_pos_bound=min(drift_test_config["max_init_pos_bound"],
                          nodrift_config["max_init_pos_bound"]),
        max_vel_bound=min(drift_test_config["max_init_vel_bound"],
                          nodrift_config["max_init_vel_bound"]),
    )
    result_a = evaluate_drift(drift_model_paths, drift_thresholds, label_a, resolved_seed,
                              shared_start_states, drift_obs_time_norms)
    result_b = evaluate_nodrift(nodrift_model_path, label_b, resolved_seed, shared_start_states)

    if EXPORT_RESULTS_CSV:
        # Records the drift-side eval config (stage 9, absent from checkpoint
        # metadata) so the CSV stays self-describing. start_state_provenance
        # is included because a loaded test set supplies its own bounds.
        _eval_fields = eval_config_export_fields(drift_test_config, start_state_provenance)
        export_episode_results_csv([result_a, result_b], BASE_DIR, "drift_vs_nodrift_test",
                                    "drift_vs_nodrift_episodes", win_outcome_value="win",
                                    extra_fields=_eval_fields)
        print()
        export_trajectory_steps_csv([result_a, result_b], BASE_DIR, "drift_vs_nodrift_test",
                                     "drift_vs_nodrift_trajectories", extra_fields=_eval_fields)

    print_comparison_table(result_a, result_b, resolved_seed)

    if PLOT_TRAJECTORIES:
        plot_comparison(result_a, result_b, resolved_seed, show=SHOW_PLOTS)
    if PLOT_DIAGNOSTICS and (result_a["trajectories"] or result_b["trajectories"]):
        plot_diagnostics(result_a, result_b, resolved_seed, show=SHOW_PLOTS)
    if PLOT_APPROACH_DIRECTIONS and (result_a["entry_directions"] or result_b["entry_directions"]):
        plot_approach_directions(result_a, result_b, resolved_seed, show=SHOW_PLOTS)
    if PLOT_FUEL_BAR:
        plot_fuel_bar(result_a, result_b, resolved_seed, show=SHOW_PLOTS)
