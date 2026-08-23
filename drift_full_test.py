"""Run a complete curriculum test in one continuous episode.

Test flow:
1. Start with the hardest curriculum model.
2. Continue until that model docks the spacecraft.
3. Switch to the next easier model.
4. Count only the final stage as the overall win or loss.

Related scripts:
- checkpoint_test.py: tests one checkpoint and one stage across several episodes.
- curriculum_evaluation.py: uses the same model-chaining approach with a fixed seed.
- nodrift_full_test.py: uses the same approach for the direct-docking agent.

Checkpoint selection:
1. Set CHECKPOINT and NUM_MODELS.
2. resolve_checkpoint_paths() finds the newest checkpoint for each stage.
3. resolve_env_config_best_effort() loads docking thresholds separately for each checkpoint.
4. To choose checkpoint files manually, set USE_MANUAL_CHECKPOINTS = True and enter their paths in MANUAL_MODEL_PATHS below.

Additional test options:
- USE_SAFE_ACTION enables the safe-action ablation.
- USE_ACTION_NOISE enables the action-noise ablation.
- The script creates a trajectory plot colored by thrust.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import drift_env
from drift_env import DriftTestEnv
from evaluation_utilities import (
    extract_stage,
    resolve_drift_model_chain,
    resolve_drift_stage_thresholds,
    resolve_drift_stage_obs_time_norm,
    resolve_drift_eval_config,
    eval_config_export_fields,
    save_and_show_figure,
    plural_word,
    build_thrust_colormap,
    max_thrust_across_results,
    interquartile_mean,
    resolve_obs_flags,
    resolve_env_flags_from_metadata,
    resolve_and_print_seed,
    write_tidy_csv,
    format_win_rate,
    detect_agent_mode,
    resolve_start_states,
    new_safe_action_counts,
    summarize_safe_action_counts,
    rollout_drift_chain_episode,
    override_dock_threshold,
    apply_dock_radius_override,
    set_equal_3d_axes,
    SAFE_ACTION_KINDS,
    SPACE_CONTROLS_FLAG_KEYS,
    DRIFT_REWARD_FLAG_KEYS,
    COAST_THRUST_THRESHOLD,
    MAX_STEPS_PER_EPISODE,
)
from stable_baselines3 import PPO

# Build the checkpoint path relative to this file so the script works
# regardless of which directory it is launched from.
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"

# Checkpoints:

# Checkpoint auto-discovery: "latest", "run:safe_PPO_20", a glob, or an exact path.
# NUM_MODELS is a generous limit so every stage loads, not just the hardest ones.
CHECKPOINT = "latest" 
NUM_MODELS = 100

# True: use the exact MANUAL_MODEL_PATHS below instead of auto-discovery
# (for mixing runs or pinning an exact chain).
USE_MANUAL_CHECKPOINTS = False

# Manual override paths, used only when USE_MANUAL_CHECKPOINTS is True.
# List hardest to easiest. Format: safe_ppo_model_{run}_{stage}_{epoch}.zip
MANUAL_MODEL_PATHS = [
    CHECKPOINT_DIR / "safe_PPO_20" / "safe_ppo_model_20_8_3.zip",  # hardest
    CHECKPOINT_DIR / "safe_PPO_20" / "safe_ppo_model_20_7_0.zip",
    CHECKPOINT_DIR / "safe_PPO_20" / "safe_ppo_model_20_6_6.zip",
    CHECKPOINT_DIR / "safe_PPO_20" / "safe_ppo_model_20_3_3.zip",
    CHECKPOINT_DIR / "safe_PPO_20" / "safe_ppo_model_20_1_0.zip",  # easiest
]

# --- Settings ---
# Grouped by what they affect; execution starts below.

# Run setup:

# Seed for episodes.
# None:   pick a random seed, printed at startup so the run is reproducible.
# An int: fix to a specific run.
SEED = None  # example: 12345

# True:  pop up a window for each plot.
# False: save PNGs to saved_figures/, no window (auto-skipped under a
#        headless Matplotlib backend).
SHOW_PLOTS = True

# Number of episodes to run.
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

# Ablations:
# These change what the agent does, so they change the results.

# Replace the model's action with a one-step lookahead safety check
# (see get_safe_action).
# True:  use the safety-checked action.
# False: use the model's raw action.
USE_SAFE_ACTION = False

# Multiply every action by random noise in [0.95, 1.05], a robustness test.
# True:  add noise.
# False: use the model's exact action.
USE_ACTION_NOISE = False

# Scale the observation's timestep element by each stage's own trained
# max_episode_len instead of the eval env's (9,000), which constrains obs[6]
# near zero all flight.
# True:  per-stage divisor, matching what each policy trained on.
# False: the eval env's own max_episode_len.
USE_PER_STAGE_OBS_TIME_NORM = True

# Win condition override:
# Force the chain's final stage to dock against a distance/speed it did
# not train with, e.g. testing this checkpoint against another agent's
# target instead of its own. 
# None: use the checkpoint's own trained thresholds, unchanged. 
# See evaluation_utilities.override_dock_threshold.
DOCKING_POSITION_OVERRIDE = None  # meters, e.g. 0.5. None: use the checkpoint's own trained value.
DOCKING_SPEED_OVERRIDE = None     # m/s, e.g. 0.2. None: use the checkpoint's own trained value.

# Console output:

# Prints per-episode index, env config, and a "Model N/9 (stage S): WIN/FAIL" line.
# True:  print all of it.
# False: only the final summary.
VERBOSE = False

# Prints thrust vector, magnitude, position, and speed each step.
# True:  print every PRINT_THRUST_EVERY steps.
# False: no per-step thrust output.
PRINT_THRUST = False
PRINT_THRUST_EVERY = 1  # print every Nth step; 1 = every step

# Saved output:

# Formatted per-episode CSV under saved_data/, ready for Seaborn.
# True:  write it (see evaluation_utilities.write_tidy_csv).
# False: skip it.
EXPORT_RESULTS_CSV = True

# 3D trajectory plot under saved_figures/.
# True:  save it.
# False: skip it.
PLOT_TRAJECTORIES = True

# How many of the first episodes to record and plot. Defaults to every
# episode; lower it if NUM_EPISODES is large and the plot gets cluttered.
NUM_PLOT_EPISODES = 10

# 2x2 per-step diagnostics figure: thrust, distance, and speed vs. time.
# True:  save it.
# False: skip it.
PLOT_DIAGNOSTICS = True

# 2x2 figure of which action the safety shield picked, and at what
# step/distance/speed margin. Ignored when USE_SAFE_ACTION is off.
# True: save it.  False: skip it.
PLOT_SAFETY_ANALYSIS = True

# First N episodes to overlay in the diagnostics figure.
DIAG_NUM_EPISODES = 5

# Figure style:

# Color each trajectory by thrust magnitude (bright = thrust, dark = coast).
# True:  color by thrust.
# False: flat win/loss color.
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
THRUST_BURST_FRACTION = 0.6

# Colormap for COLOR_TRAJECTORY_BY_THRUST. All but "grey_to_warm" are
# colorblind-safe.
#   "discrete"           Default. Grey (coast) to blue (thrust), 0.25 N bands.
#   "cividis"/"viridis"  Continuous, perceptually uniform.
#   "grey_to_warm"       High contrast, NOT colorblind-safe.
THRUST_COLORMAP = "discrete"

# Where the 3D plot's thrust colorbar sits.
# True:  horizontal bar below the axes.
# False: vertical bar at the right (Matplotlib's default).
THRUST_COLORBAR_AT_BOTTOM = False

# Shade each diagnostics line with its own tint of blue (win) or red (loss).
# True:  per-episode gradient.
# False: one flat color per outcome.
DIAGNOSTICS_GRADIENT_BY_EPISODE = True

# COAST_THRUST_THRESHOLD is imported from evaluation_utilities, not set here,
# so this script's thrust-panel reference line matches the coast fraction
# other scripts compute with it.

# --- Execution ---
# Seed, checkpoint discovery, model loading, then the episode loop.
# Configure using the settings above, not here.

# Resolve the seed BEFORE any PPO.load() call below: PPO.load() reseeds
# NumPy's global RNG, so a "random" seed drawn after it is not
# random.
if __name__ == "__main__":
    resolved_seed = resolve_and_print_seed(SEED, leading_newline=True)

    if USE_MANUAL_CHECKPOINTS:
        model_paths = [Path(p) for p in MANUAL_MODEL_PATHS]
    else:
        # Returns the stage models hardest-first, which is the order the
        # episode loop below replays them in.
        model_paths = resolve_drift_model_chain(CHECKPOINT, CHECKPOINT_DIR, NUM_MODELS)

    # Catch a nodrift checkpoint early (folder typo or incorrect manual path)
    # instead of failing deep inside DriftTestEnv with a confusing error.
    # Checks the whole chain since MANUAL_MODEL_PATHS can mix folders.
    _wrong_family = [p for p in model_paths if detect_agent_mode(str(p)) != "drift"]
    if _wrong_family:
        raise ValueError(
            f"CHECKPOINT/MANUAL_MODEL_PATHS resolved to {len(_wrong_family)} "
            f"checkpoint(s) not detected as drift: {[str(p) for p in _wrong_family]}. "
            f"This script only supports drift checkpoints."
        )

    models = [PPO.load(str(p)) for p in model_paths]
    print(f"Chain: {len(model_paths)} models, hardest {model_paths[0].name}")

    # resolve_drift_eval_config() is the drift test env config, the bounds
    # this chain is evaluated under.
    _test_config, _ = resolve_drift_eval_config()
    start_states, start_state_provenance = resolve_start_states(
        TEST_SET, NUM_EPISODES, resolved_seed,
        min_pos_bound=_test_config['min_init_pos_bound'],
        max_pos_bound=_test_config['max_init_pos_bound'],
        max_vel_bound=_test_config['max_init_vel_bound'],
        leading_newline=True,
    )

    # Match the eval env to training config. Obs-extending flags come from
    # observation size; reward-side flags come from run_metadata.json.
    # space_controls_obs is read first since it changes the size-to-flags mapping.
    SPACE_CONTROLS_FLAGS = resolve_env_flags_from_metadata(
        str(model_paths[0]), keys=SPACE_CONTROLS_FLAG_KEYS
    )
    SAFERL_OBS, CUSTOM_BRAKING_MARGIN_OBS = resolve_obs_flags(
        models[0].observation_space.shape[0],
        space_controls_obs=SPACE_CONTROLS_FLAGS.get("space_controls_obs", False),
    )
    DRIFT_ENV_FLAGS = {
        "saferl_obs": SAFERL_OBS,
        "custom_braking_margin_obs": CUSTOM_BRAKING_MARGIN_OBS,
        **resolve_env_flags_from_metadata(
            str(model_paths[0]),
            keys=DRIFT_REWARD_FLAG_KEYS,
        ),
        **SPACE_CONTROLS_FLAGS,
    }
    _non_default = {k: v for k, v in DRIFT_ENV_FLAGS.items() if v}
    if _non_default:
        print(f"Env flags (from checkpoint): {_non_default}")

    # Run folder name for the hardest model, used to label the trajectory plot.
    # Reflects only the hardest model's folder if paths mix folders.
    RUN_FOLDER_NAME = model_paths[0].parent.name

    # Docking thresholds per model, same order as model_paths (hardest first).
    thresholds = [resolve_drift_stage_thresholds(str(p)) for p in model_paths]

    print()
    print("Models (chain order: farthest start first, tightest final dock last):")
    for path, (pos_thresh, speed_thresh) in zip(model_paths, thresholds):
        print(f"  stage {extract_stage(str(path)):<3} {path.name}  "
              f"(pos_thresh={pos_thresh}, speed_thresh={speed_thresh})")
    print()
    print(f"{plural_word(NUM_EPISODES, 'Episode')}: {NUM_EPISODES}")

    # Per-stage observation timestep divisors, same order as model_paths.
    # None keeps the eval env's own max_episode_len.
    STAGE_OBS_TIME_NORMS = (
        [resolve_drift_stage_obs_time_norm(str(p)) for p in model_paths]
        if USE_PER_STAGE_OBS_TIME_NORM else None
    )
    if STAGE_OBS_TIME_NORMS is not None:
        print(f"Per-stage obs timestep divisors: {STAGE_OBS_TIME_NORMS}")

    if DOCKING_POSITION_OVERRIDE is not None or DOCKING_SPEED_OVERRIDE is not None:
        _trained_pos = apply_dock_radius_override(
            thresholds[-1][0], SPACE_CONTROLS_FLAGS, "drift")
        override_dock_threshold(
            _trained_pos, thresholds[-1][1],
            DOCKING_POSITION_OVERRIDE, DOCKING_SPEED_OVERRIDE, label="Chain",
        )


    def plot_full_trajectories(trajectories, thrusts, outcomes, num_episodes, run_folder,
                               seed=None, show=True):
        """Plot 3D trajectories of full-range chain episodes and save to disk.

        Each trajectory covers the full chain, farthest start to final dock.
        With COLOR_TRAJECTORY_BY_THRUST, segments are colored by thrust
        (dark = coast, bright = thrust); otherwise flat blue (win) or red (loss).

        Args:
            trajectories: List of trajectories, one per episode. Each trajectory
                is a list of (x, y, z) positions recorded every step.
            thrusts: List of per-step thrust magnitudes, parallel to
                trajectories (thrusts[i][j] corresponds to trajectories[i][j]).
            outcomes: List of outcome strings, one per episode.
            num_episodes: Total number of episodes run, used in the title.
            run_folder: Name of the checkpoint run folder (e.g. "safe_PPO_16"),
                shown in the title so the plot can be traced back to its models.
            seed: The resolved seed, noted in the title so the figure records
                which run produced it.
            show: False saves the PNG without popping up a window (also
                skipped automatically under a headless Matplotlib backend).
        """
        fig = plt.figure(figsize=(13, 9.5))
        ax = fig.add_subplot(111, projection="3d")

        # Color blind-friendly palette. Blue for wins, red for losses.
        WIN_COLOR = "#005AB5"
        LOSS_COLOR = "#DC3220"

        win_plotted = loss_plotted = burst_labeled = False
        thrust_line = None  # last Line3DCollection drawn, used to anchor the colorbar

        if COLOR_TRAJECTORY_BY_THRUST:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            # One shared color scale across all episodes, so the same shade
            # means the same thrust between trajectories, not just within one.
            max_thrust = max_thrust_across_results({"thrusts": thrusts})
            thrust_cmap, thrust_norm = build_thrust_colormap(THRUST_COLORMAP, max_thrust)

        for traj, thrust, outcome in zip(trajectories, thrusts, outcomes):
            traj = np.array(traj)
            if len(traj) < 2:
                continue

            is_win = outcome == "win"
            color = WIN_COLOR if is_win else LOSS_COLOR

            if COLOR_TRAJECTORY_BY_THRUST:
                # Build one small line segment per step and color it by the
                # thrust magnitude used on that step (bright = hard thrust).
                points = traj.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_thrust = np.array(thrust[1:len(traj)])
                lc = Line3DCollection(segments, cmap=thrust_cmap, norm=thrust_norm,
                                       alpha=0.9)
                lc.set_array(segment_thrust)
                if THRUST_BURST_MARKERS:
                    # Thicker line for harder thrust, not just a color change,
                    # so a brief hard thrust stands out.
                    lc.set_linewidth(1.5 + 4.5 * (segment_thrust / max_thrust))
                else:
                    lc.set_linewidth(2.0)
                ax.add_collection3d(lc)
                thrust_line = lc

                if THRUST_BURST_MARKERS:
                    # Triangle markers on steps thrusting hard enough, so a brief
                    # thrust stands out instead of blending into the line.
                    burst_mask = segment_thrust >= THRUST_BURST_FRACTION * max_thrust
                    if np.any(burst_mask):
                        burst_points = traj[1:][burst_mask]
                        burst_values = segment_thrust[burst_mask]
                        burst_label = None
                        if not burst_labeled:
                            burst_label = "Hard thrust"
                            burst_labeled = True
                        # Colored by thrust magnitude like the line itself;
                        # BRIGHT_THRUST_HIGHLIGHTS just controls marker size/opacity.
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

                # Still mark win/loss so the legend and starting dot are meaningful.
                label = None
                if is_win and not win_plotted:
                    label = "Win (start)"
                    win_plotted = True
                elif not is_win and not loss_plotted:
                    label = "Loss (start)"
                    loss_plotted = True
                ax.scatter(*traj[0], color=color, s=25, alpha=0.9,
                           edgecolors="black", linewidths=0.5, label=label)
            else:
                label = None
                if is_win and not win_plotted:
                    label = "Win"
                    win_plotted = True
                elif not is_win and not loss_plotted:
                    label = "Loss"
                    loss_plotted = True

                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        color=color, linewidth=1.2, alpha=0.6, label=label)
                # Mark the starting position with a dot.
                ax.scatter(*traj[0], color=color, s=20, alpha=0.9)

        if thrust_line is not None:
            if THRUST_COLORBAR_AT_BOTTOM:
                # shrink/pad tuned so the horizontal bar doesn't overlap the axes' bottom edge/tick labels.
                cbar = fig.colorbar(thrust_line, ax=ax, location="bottom", orientation="horizontal",
                                    shrink=0.4, pad=0.05)
            else:
                cbar = fig.colorbar(thrust_line, ax=ax, location="right", orientation="vertical",
                                    shrink=0.6, pad=0.04)
            cbar.set_label("Thrust magnitude (Newtons, L2 norm per step)", labelpad=10)

        # Chief spacecraft is always at the origin.
        ax.scatter(0, 0, 0, marker="x", color="black", s=150, linewidths=2.5,
                   zorder=5, label="Chief (target)")

        # Force equal, symmetric axis ranges so the chief sits at the plot's center.
        all_points = np.concatenate([np.array(t) for t in trajectories if len(t) > 0])
        set_equal_3d_axes(ax, all_points)
        seed_note = f" (seed {seed})" if seed is not None else ""
        ax.set_title(f"Drifter Full-Range Chain Trajectories - {run_folder}{seed_note}\n"
                     f"{num_episodes} {plural_word(num_episodes, 'Episode')}, Win rate: "
                     f"{sum(1 for o in outcomes if o == 'win')}/{num_episodes}")
        ax.legend(loc="upper left")
        plt.tight_layout()
        # tight_layout() alone can still let a tall 3D box's title clip off
        # the top of the window; reserve explicit headroom for it.
        fig.subplots_adjust(top=0.92)

        save_and_show_figure(BASE_DIR, "drift_full_test",
                             "full_chain_trajectories", "trajectory plot", show=show)


    def plot_diagnostics(trajectories, thrusts, speeds, outcomes, run_folder, seed=None,
                         show=True):
        """Save a 2x2 per-step diagnostics figure for the chained episodes.

        Panels: thrust vs. time, thrust vs. distance to the chief, distance
        vs. time, and speed vs. time, one line per episode (blue win, red
        loss), for the first DIAG_NUM_EPISODES episodes. The 3D plot shows
        where the agent flies; this shows how, so drift/coast stretches read
        as long near-zero thrust runs. Time is in steps, which are 1 s each.

        Args:
            trajectories: List of per-episode (x, y, z) position lists.
            thrusts: Per-step thrust magnitude, parallel to trajectories.
            speeds: Per-step speed, parallel to trajectories.
            outcomes: "win"/"loss" per episode.
            run_folder: Checkpoint run folder name, shown in the title.
            seed: The resolved seed, noted in the title so the figure records
                which run produced it.
            show: False saves the PNG without popping up a window (also
                skipped automatically under a headless Matplotlib backend).
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        (ax_thrust_time, ax_thrust_dist), (ax_dist_time, ax_speed_time) = axes

        WIN_COLOR = "#005AB5"
        LOSS_COLOR = "#DC3220"
        win_labeled = loss_labeled = False

        n_episodes = min(DIAG_NUM_EPISODES, len(trajectories))

        if DIAGNOSTICS_GRADIENT_BY_EPISODE:
            # Rank each episode within its own outcome group so shades stay distinct.
            # Gradients run past their pure hue into a neighbor (blue-purple,
            # red-orange) to stay distinguishable in groups of 5+.
            from matplotlib.colors import LinearSegmentedColormap
            win_cmap = LinearSegmentedColormap.from_list(
                "win_blue_purple",
                [(0.0, "#7EC8E3"), (0.55, "#005AB5"), (0.8, "#0B3D91"), (1.0, "#5A189A")],
            )
            loss_cmap = LinearSegmentedColormap.from_list(
                "loss_red_orange", ["#F4A261", "#DC3220", "#A61C00", "#7A1300"]
            )
            n_wins = sum(1 for i in range(n_episodes) if outcomes[i] == "win")
            n_losses = n_episodes - n_wins
            win_rank = loss_rank = 0

        for i in range(n_episodes):
            traj = np.array(trajectories[i])
            thrust = np.array(thrusts[i])
            speed = np.array(speeds[i])
            if len(traj) < 2:
                continue
            distance = np.linalg.norm(traj, axis=1)
            time = np.arange(len(traj))

            is_win = outcomes[i] == "win"
            if DIAGNOSTICS_GRADIENT_BY_EPISODE:
                if is_win:
                    shade = win_rank / max(n_wins - 1, 1)
                    color = win_cmap(shade)
                    win_rank += 1
                else:
                    shade = loss_rank / max(n_losses - 1, 1)
                    color = loss_cmap(shade)
                    loss_rank += 1
            else:
                color = WIN_COLOR if is_win else LOSS_COLOR
            label = None
            if is_win and not win_labeled:
                label = "Win"
                win_labeled = True
            elif not is_win and not loss_labeled:
                label = "Loss"
                loss_labeled = True

            ax_thrust_time.plot(time, thrust, color=color, alpha=0.5, linewidth=1.0, label=label)
            ax_thrust_dist.plot(distance, thrust, color=color, alpha=0.5, linewidth=1.0)
            ax_dist_time.plot(time, distance, color=color, alpha=0.5, linewidth=1.0)
            ax_speed_time.plot(time, speed, color=color, alpha=0.5, linewidth=1.0)

        # Coast-threshold reference line on both thrust panels.
        for ax in (ax_thrust_time, ax_thrust_dist):
            ax.axhline(COAST_THRUST_THRESHOLD, color="black", linestyle=":",
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
            f"Drifter per-step diagnostics - {run_folder}{seed_note}\n"
            f"First {n_episodes} {plural_word(n_episodes, 'episode')}\n"
            f"Dotted line = coast threshold {COAST_THRUST_THRESHOLD} Newtons",
            fontsize=11,
        )
        plt.tight_layout()

        save_and_show_figure(BASE_DIR, "drift_full_test",
                             "drift_diagnostics", "diagnostics plot", show=show)


    def plot_safety_analysis(events, run_folder, seed=None, show=True):
        """Save a 2x2 figure describing when and where the safety check acted.

        Only meaningful with USE_SAFE_ACTION on; the events list is empty
        otherwise. The four panels show which candidate the shield picked,
        when in the episode it acted, at what distance, and how close to the
        speed limit the deputy was.

        The distance panel is the useful one for a chained run: distance
        stands in for curriculum stage, so a shield that only fires beyond
        some range only fires for the earlier, weaker models in the chain.

        Args:
            events: Rows from get_safe_action()'s events list, one per
                checked step.
            run_folder: Checkpoint run folder name, shown in the title.
            seed: The resolved seed, noted in the title.
            show: False saves the PNG without popping up a window.
        """
        overrides = [e for e in events if e["overridden"]]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        (ax_kind, ax_step), (ax_dist, ax_margin) = axes

        # Panel 1: which candidate was chosen, over every checked step.
        kinds = list(SAFE_ACTION_KINDS)
        kind_counts = [sum(1 for e in events if e["kind"] == k) for k in kinds]
        bars = ax_kind.bar(kinds, kind_counts, color="#0072B2", alpha=0.85,
                           edgecolor="black", linewidth=0.5)
        # "model" bar means the policy was already safe; color it differently.
        if bars:
            bars[0].set_color("#009E73")
        for bar, count in zip(bars, kind_counts):
            if count:
                ax_kind.annotate(f"{100 * count / len(events):.1f}%",
                                 xy=(bar.get_x() + bar.get_width() / 2, count),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha="center", fontsize=9)
        ax_kind.set_ylabel(f"Checked steps (of {len(events)})")
        ax_kind.set_title("Which action the safety check used\n"
                          "(green = policy was already safe, no override)", fontsize=10)
        ax_kind.grid(axis="y", alpha=0.3)

        if not overrides:
            # A shield that never fires is a real result, not an empty plot.
            for ax, title in ((ax_step, "When the shield acted"),
                              (ax_dist, "Distance when the shield acted"),
                              (ax_margin, "Speed vs limit when the shield acted")):
                ax.text(0.5, 0.5,
                        "The shield never overrode the policy.\n"
                        "Every checked step was already safe,\n"
                        "so there is nothing to distribute here.",
                        ha="center", va="center", fontsize=11, color="#444444",
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Panel 2: when in the episode overrides happened.
            ax_step.hist([e["step"] for e in overrides], bins=30,
                         color="#D55E00", alpha=0.85, edgecolor="black", linewidth=0.4)
            ax_step.set_xlabel("Step within the episode")
            ax_step.set_ylabel("Overrides")
            ax_step.set_title("When the shield acted", fontsize=10)
            ax_step.grid(alpha=0.3)

            # Panel 3: distance at override, split by which candidate was used,
            # so "zero thrust far out" and "braking up close" separate visibly.
            for kind, color in (("zero_thrust", "#E69F00"), ("braking", "#CC79A7"),
                                ("none_safe", "#D55E00")):
                values = [e["distance"] for e in overrides if e["kind"] == kind]
                if values:
                    ax_dist.hist(values, bins=30, alpha=0.6, label=f"{kind} (n={len(values)})",
                                 color=color, edgecolor="black", linewidth=0.3)
            ax_dist.set_xlabel("Distance from chief (m)")
            ax_dist.set_ylabel("Overrides")
            ax_dist.set_title("Distance when the shield acted\n"
                              "(distance stands in for stage in a chain)", fontsize=10)
            ax_dist.legend(fontsize=8)
            ax_dist.grid(alpha=0.3)

            # Panel 4: how close to the speed limit the deputy already was.
            # Zero is exactly at the limit; positive is already violating.
            margins = [e["speed_margin"] for e in overrides]
            ax_margin.hist(margins, bins=30, color="#0072B2", alpha=0.85,
                           edgecolor="black", linewidth=0.4)
            ax_margin.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
                              label="at the speed limit")
            ax_margin.set_xlabel("Speed minus limit (m/s), before the shield acted")
            ax_margin.set_ylabel("Overrides")
            ax_margin.set_title("Speed vs limit when the shield acted", fontsize=10)
            ax_margin.legend(fontsize=8)
            ax_margin.grid(alpha=0.3)

        seed_note = f" (seed {seed})" if seed is not None else ""
        override_pct = 100 * len(overrides) / len(events) if events else 0.0
        fig.suptitle(
            f"Safety check (USE_SAFE_ACTION) analysis - {run_folder}{seed_note}\n"
            f"{len(overrides)} of {len(events)} checked "
            f"{plural_word(len(events), 'step')} overridden ({override_pct:.1f}%)",
            fontsize=12,
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)

        save_and_show_figure(BASE_DIR, "drift_full_test",
                             "drift_safety_analysis", "safety analysis plot", show=show)


    fuels = []      # per-episode L2 Delta-v (gimbaled equivalent; secondary)
    fuels_l1 = []   # per-episode L1 Delta-v (propellant used; primary)
    wins = []       # 1 for a successful dock, 0 for a failure
    episode_lengths = []    # total step count across the whole chain, per episode
    elapsed_seconds_list = []  # real simulated time across the whole chain, per episode
    failure_counts = {
        "crash": 0, "unsafe": 0, "out_of_bounds": 0, "fuel": 0,
        "timeout": 0, "unknown": 0,
    }
    all_trajectories = []   # full position trajectory per episode, if plotting
    all_thrusts = []        # per-step thrust magnitude, parallel to all_trajectories
    all_speeds = []         # per-step speed, parallel to all_trajectories
    all_outcomes = []       # win or loss per episode, if plotting
    # Which candidate the safety check picked, when USE_SAFE_ACTION is on.
    safe_action_counts = new_safe_action_counts()
    # One row per checked step: which candidate won, and the distance/speed
    # it was decided at. Empty when USE_SAFE_ACTION is off.
    safe_action_events = []

    for i in range(NUM_EPISODES):
        print(f'Episode {i + 1}/{NUM_EPISODES}')
        if VERBOSE:
            print(f'starting at model 0 (hardest, {len(models)} total)')

        # Create a fresh curriculum and environment for each episode.
        # verbose=False: already printed once above, before the episode loop.
        curriculum, _ = resolve_drift_eval_config(verbose=False)
        env = DriftTestEnv(**DRIFT_ENV_FLAGS, **curriculum)
        # Override the env's own sampling so every run faces the same starts.
        env.env.fixed_start = True
        env.env.fixed_state = start_states[i].copy()
        obs, info = env.reset(seed=resolved_seed + i)

        ep_result = rollout_drift_chain_episode(
            env, obs, models, thresholds, model_paths,
            max_steps=MAX_STEPS_PER_EPISODE,
            use_safe_action=USE_SAFE_ACTION,
            safe_action_counts=safe_action_counts,
            safe_action_events=safe_action_events,
            episode_idx=i,
            use_action_noise=USE_ACTION_NOISE,
            print_thrust=PRINT_THRUST,
            print_thrust_every=PRINT_THRUST_EVERY,
            verbose=VERBOSE,
            stage_obs_time_norms=STAGE_OBS_TIME_NORMS,
            final_pos_override=DOCKING_POSITION_OVERRIDE,
            final_speed_override=DOCKING_SPEED_OVERRIDE,
        )

        wins.append(1 if ep_result["win"] else 0)
        fuels.append(ep_result["fuel"])
        fuels_l1.append(ep_result["fuel_l1"])
        episode_lengths.append(ep_result["episode_length"])
        elapsed_seconds_list.append(ep_result["elapsed_seconds"])
        if ep_result["failure_type"] is not None:
            failure_counts[ep_result["failure_type"]] += 1

        # Record the trajectory for this episode if either plot needs it.
        if (PLOT_TRAJECTORIES or PLOT_DIAGNOSTICS) and i < NUM_PLOT_EPISODES:
            all_trajectories.append(ep_result["positions"])
            all_thrusts.append(ep_result["thrusts"])
            all_speeds.append(ep_result["speeds"])
            all_outcomes.append(ep_result["outcome"])

    # Print a summary of results across all episodes.
    print()
    print(f"Results over {len(wins)} {plural_word(len(wins), 'episode')}:")
    print(f'  Win rate:  {format_win_rate(sum(wins), len(wins))}')
    print(f'  Mean episode length: {np.mean(episode_lengths):.1f} steps '
          f'({np.mean(elapsed_seconds_list):.0f}s real elapsed time, median '
          f'{np.median(elapsed_seconds_list):.0f}s)')

    top_failure = max(failure_counts, key=failure_counts.get)
    top_count = failure_counts[top_failure]
    top_failure_str = "N/A" if top_count == 0 else f"{top_failure} x{top_count}"
    print(f'  Top failure: {top_failure_str}')

    # Only meaningful when the safety check ran.
    if USE_SAFE_ACTION:
        _safe = summarize_safe_action_counts(safe_action_counts)
        print()
        print(f"Safe-action check over {_safe['safe_action_checked_steps']} checked "
              f"{plural_word(_safe['safe_action_checked_steps'], 'step')}:")
        print(f"  Model action already safe:  {100 * _safe['safe_action_model_fraction']:.1f}%")
        print(f"  Overridden to zero thrust:  {100 * _safe['safe_action_zero_thrust_fraction']:.1f}%")
        print(f"  Overridden to braking:      {100 * _safe['safe_action_braking_fraction']:.1f}%")
        print(f"  No safe candidate found:    {100 * _safe['safe_action_none_safe_fraction']:.1f}%")

    # L1 is the real fuel cost (propellant the fixed thrusters use), reported
    # first. L2 (net velocity change) is a secondary metric, reported second.
    print()
    print(f'Delta-v, per-axis sum (L1, m/s)  [propellant cost]:')
    print(f'  Mean:      {np.mean(fuels_l1):.3f}')
    print(f'  Median:    {np.median(fuels_l1):.3f}')
    print(f'  IQM:       {interquartile_mean(fuels_l1):.3f}')
    print(f'  Min:       {min(fuels_l1):.3f}')
    print(f'  Max:       {max(fuels_l1):.3f}')
    print(f'  25th pct:  {np.percentile(fuels_l1, 25):.3f}')
    print(f'  50th pct:  {np.percentile(fuels_l1, 50):.3f}')
    print(f'  75th pct:  {np.percentile(fuels_l1, 75):.3f}')

    print()
    print(f'Delta-v, vector norm (L2, m/s)  [gimbaled equivalent]:')
    print(f'  Mean:      {np.mean(fuels):.3f}')
    print(f'  Median:    {np.median(fuels):.3f}')
    print(f'  IQM:       {interquartile_mean(fuels):.3f}')
    print(f'  Min:       {min(fuels):.3f}')
    print(f'  Max:       {max(fuels):.3f}')
    print(f'  25th pct:  {np.percentile(fuels, 25):.3f}')
    print(f'  50th pct:  {np.percentile(fuels, 50):.3f}')
    print(f'  75th pct:  {np.percentile(fuels, 75):.3f}')
    print()

    if EXPORT_RESULTS_CSV:
        # Create one Seaborn-ready row per episode. Include the stage 9
        # evaluation settings and the test file's start-state source so the
        # CSV accurately documents this run, even if settings change later.
        _eval_fields = eval_config_export_fields(_test_config, start_state_provenance)
        rows = [
            {
                "condition": RUN_FOLDER_NAME,
                "episode_idx": i,
                "win": bool(wins[i]),
                "fuel_l2": fuels[i],
                "fuel_l1": fuels_l1[i],
                "episode_length": episode_lengths[i],
                "elapsed_seconds": elapsed_seconds_list[i],
                **_eval_fields,
            }
            for i in range(len(wins))
        ]
        # Save with a timestamp so repeated runs do not overwrite older CSV files.
        # This matches the other evaluation scripts and keeps DRIFT ablations easy to compare.
        _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_tidy_csv(rows, BASE_DIR / "saved_data" / "drift_full_test" / f"drift_full_episodes_{RUN_FOLDER_NAME}_{_timestamp}.csv",
                       data_label="episode results")

    # Plot trajectories (if enabled)
    if PLOT_TRAJECTORIES and all_trajectories:
        plot_full_trajectories(all_trajectories, all_thrusts, all_outcomes, len(wins),
                               RUN_FOLDER_NAME, seed=resolved_seed, show=SHOW_PLOTS)

    # Per-step diagnostics (if enabled)
    if PLOT_DIAGNOSTICS and all_trajectories:
        plot_diagnostics(all_trajectories, all_thrusts, all_speeds, all_outcomes,
                         RUN_FOLDER_NAME, seed=resolved_seed, show=SHOW_PLOTS)

    # Safety-check analysis (only when the check ran)
    if PLOT_SAFETY_ANALYSIS and USE_SAFE_ACTION and safe_action_events:
        plot_safety_analysis(safe_action_events, RUN_FOLDER_NAME,
                             seed=resolved_seed, show=SHOW_PLOTS)
