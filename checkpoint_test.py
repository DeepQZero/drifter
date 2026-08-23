"""Evaluate one trained checkpoint over several episodes.

Reports:
- Win rate and failure reasons
- A 3D trajectory plot
- A JSON results file
- Drift and nodrift checkpoint support

Set AGENT_MODE to drift, nodrift, or auto. Auto detects the mode from
the filename.

Best for NODRIFT checkpoints. One DRIFT stage is only part of a chained
curriculum, not a complete docking policy. Use drift_full_test.py for a
full drift evaluation.

Fuel metrics:
1. Delta-v, per-axis sum (L1): primary metric for three fixed thrusters.
2. Delta-v, vector norm (L2): secondary metric for one gimbaled thruster.
3. Action magnitude: raw action size, not a fuel metric; kept for older logs.
"""

import os
import json
import argparse
import contextlib
import io
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import drift_env
import docking_env
from evaluation_utilities import (
    resolve_checkpoint_paths,
    resolve_agent_mode,
    patch_unsafe_termination,
    classify_failure,
    classify_failure_nodrift,
    resolve_obs_flags,
    resolve_env_flags_from_metadata,
    resolve_env_config_best_effort,
    center_matplotlib_window,
    should_show_plot,
    plural_word,
    interquartile_mean,
    resolve_and_print_seed,
    write_tidy_csv,
    format_win_rate,
    set_equal_3d_axes,
    SPACE_CONTROLS_FLAG_KEYS,
    NODRIFT_REWARD_FLAG_KEYS,
    DRIFT_REWARD_FLAG_KEYS,
    MAX_STEPS_PER_EPISODE,
)

# --- Settings ---
# Grouped by what they affect; execution starts below.

# Checkpoints:

# Which checkpoint to evaluate.
#   "latest"                       newest .zip in the newest run folder
#   "run:<folder>" or a folder     newest .zip in that folder
#   "pattern:<glob>"               custom glob, e.g. *_6_*.zip for stage 6
#   an exact .zip path             that one checkpoint
CHECKPOINT = "latest"

# Only used when CHECKPOINT is "latest": restricts which run folders count,
# so "latest" tracks the newest drift or nodrift run specifically instead
# of whichever folder is newest on disk.
LATEST_FOLDER_PREFIX = "nodrift"  # "drift" or "nodrift"

# Which environment to build.
#   "drift"    DriftTestEnv
#   "nodrift"  SpaceCraftDockingEnv3D
#   "auto"     detect from the filename ("nodrift" in name -> nodrift)
AGENT_MODE = "auto"

# Run setup:

# Number of episodes to run.
NUM_EPISODES = 30

# Seed for episodes.
# None:   pick a random seed, printed at startup so the run is reproducible.
# An int: fix to a specific run.
SEED = None  # example: 12345

# True:  display a window for each plot.
# False: save the PNG to saved_figures/ and skip the window.
SHOW_PLOTS = True

# Environment config:

# Where the eval environment config comes from.
# True:  resolve it from the checkpoint's own run_metadata.json, so it
#        always matches what the model trained on. Falls back to
#        ENV_CONFIG below (with a warning) for checkpoints with no metadata.
# False: always use ENV_CONFIG below.
AUTO_ENV_CONFIG = True

# Manual fallback config, used only when AUTO_ENV_CONFIG is False or the
# checkpoint has no metadata. Should match the trained stage (values live
# in get_curriculum()). The example below is a drift curriculum stage.
ENV_CONFIG = {
    "min_init_pos_bound": 100,
    "max_init_pos_bound": 150,
    "max_init_vel_bound": 0.5,
    "max_boundary_box": 200,
    "max_episode_len": 10,
    "max_lookahead_len": 50,
    "drift_step_len": 10,
    "pos_thresh": 5,
    "speed_thresh": 0.4,
    "max_total_dv": 1000.0,
}

# True:  exact closed-form CWH propagation instead of solve_ivp. Matches
#        what the trainers use, and roughly 8x faster; agrees with
#        solve_ivp to about 1e-11 m, far below any dock threshold.
# False: solve_ivp (RK45), the same integrator the environments used to
#        default to. Use this to check a result against the original solver.
FAST_ANALYTIC_PROPAGATION = True
drift_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION
docking_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION

# Drift only: resets the timestep counter (state[6]) to stay in the 5-10
# range drift stages trained on. Kept only in this script, and makes its 
# drift numbers non-comparable to the other eval scripts.
DRIFT_TIMESTEP_CAP = 10

# Visual override:
# Widen the start range and episode length for a longer trajectory plot.
# Nodrift only since drift stages train on short episode lengths (5-10 steps), 
# so widening to 3,000 steps pushes obs[6] out of distribution.
# True:  apply VISUAL_OVERRIDE_CONFIG below.
# False: use the checkpoint's trained config.
VISUAL_RANGE_OVERRIDE = False

# Applied when VISUAL_RANGE_OVERRIDE is True. max_lookahead_len is
# drift-only (SpaceCraftDockingEnv3D does not have this parameter), so 
# it is added conditionally in make_env() rather than included here.
VISUAL_OVERRIDE_CONFIG = {
    "min_init_pos_bound": 100,
    "max_init_pos_bound": 150,
    "max_boundary_box": 200,
    "max_episode_len": 3_000,
}
VISUAL_OVERRIDE_MAX_LOOKAHEAD_LEN = 300  # drift only

# Saved output:
# Formatted per-episode CSV under saved_data/, alongside the JSON summary.
# True:  write it (see evaluation_utilities.write_tidy_csv).
# False: skip it.
EXPORT_RESULTS_CSV = True

# Folder this script lives in, used to build paths to checkpoints,
# saved figures, and saved results regardless of where the script is
# launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def resolve_env_config(model_path: str) -> tuple[dict, str, bool]:
    """Resolve which environment config to evaluate a checkpoint with.

    When AUTO_ENV_CONFIG is on, tries the checkpoint's own
    run_metadata.json first, then a live get_curriculum() lookup where
    that is safe (any drift checkpoint, or nodrift stage 0-3). Falls
    back to the manual ENV_CONFIG below, marked unverified, only when
    neither source resolves anything.

    Args:
        model_path: Full path to the checkpoint file being evaluated.

    Returns:
        A (config, source, verified) tuple; config is a dict ready to
        pass as **kwargs to the matching environment class.
    """
    if AUTO_ENV_CONFIG:
        resolved, source, verified = resolve_env_config_best_effort(
            model_path, fallback_config=ENV_CONFIG
        )
        print(f"Env config: {source}")
        # fixed_state is stored as a string in run_metadata.json.
        # Drop it here; fixed_start remains False during evaluation.
        resolved.pop("fixed_state", None)
        return resolved, source, verified
    return ENV_CONFIG, "manual ENV_CONFIG (AUTO_ENV_CONFIG is off)", False


def make_env(mode: str, expected_obs_size: int = 7, model_path: str = None):
    """Create the evaluation environment for the given agent mode.

    Args:
        mode: "drift" for DriftTestEnv, "nodrift" for SpaceCraftDockingEnv3D.
        expected_obs_size: The loaded model's observation_space size,
            used to match the nodrift env's obs flags (see
            resolve_obs_flags). Ignored for drift.
        model_path: Full path to the checkpoint, used to auto-resolve
            the environment config. Required when AUTO_ENV_CONFIG is True.

    Returns:
        A (env, config_source, config_verified) tuple.

    Raises:
        ValueError: If mode is not "drift" or "nodrift".
    """
    env_config, source, verified = resolve_env_config(model_path)

    if VISUAL_RANGE_OVERRIDE:
        env_config = dict(env_config)
        env_config.update(VISUAL_OVERRIDE_CONFIG)
        if mode == "drift":
            env_config["max_lookahead_len"] = VISUAL_OVERRIDE_MAX_LOOKAHEAD_LEN
        print("Env config widened via VISUAL_RANGE_OVERRIDE.")

    # space_controls_obs must be read BEFORE resolve_obs_flags: it subtracts an
    # observation element, so size alone does not identify flags.
    space_controls_flags = resolve_env_flags_from_metadata(model_path, keys=SPACE_CONTROLS_FLAG_KEYS)
    space_controls_obs = space_controls_flags.get("space_controls_obs", False)
    try:
        saferl_obs, custom_braking_margin_obs = resolve_obs_flags(
            expected_obs_size, space_controls_obs=space_controls_obs
        )
    except ValueError:
        # Without run_metadata.json, space_controls_obs defaults to False.
        # If the size is unreachable, try the only other valid setting.
        flipped = not space_controls_obs
        saferl_obs, custom_braking_margin_obs = resolve_obs_flags(
            expected_obs_size, space_controls_obs=flipped
        )
        space_controls_flags["space_controls_obs"] = flipped
        print(
            f"  NOTE: obs size {expected_obs_size} is unreachable with "
            f"space_controls_obs={space_controls_obs} (the default for a "
            f"checkpoint with no recorded value); using "
            f"space_controls_obs={flipped} instead, the only value that "
            f"reaches this size."
        )
    # Reward-shape flags don't change observation size, so they must be read
    # from metadata explicitly, or a checkpoint runs under the wrong reward.
    if mode == "drift":
        from drift_env import DriftTestEnv
        reward_flags = resolve_env_flags_from_metadata(model_path, keys=DRIFT_REWARD_FLAG_KEYS)
        env = DriftTestEnv(
            saferl_obs=saferl_obs,
            custom_braking_margin_obs=custom_braking_margin_obs,
            **space_controls_flags,
            **reward_flags,
            **env_config,
        )
    elif mode == "nodrift":
        from docking_env import SpaceCraftDockingEnv3D
        reward_flags = resolve_env_flags_from_metadata(model_path, keys=NODRIFT_REWARD_FLAG_KEYS)
        env = SpaceCraftDockingEnv3D(
            saferl_obs=saferl_obs,
            custom_braking_margin_obs=custom_braking_margin_obs,
            **space_controls_flags,
            **reward_flags,
            **env_config,
        )
    else:
        raise ValueError(f"Unknown AGENT_MODE: {mode!r}. Use 'drift', 'nodrift', or 'auto'.")

    return env, source, verified


def check_win(env, mode: str) -> tuple[bool, str]:
    """Determine the outcome of a completed episode.

    For drift agents, also checks whether the agent ended in a position
    where drifting would complete the dock (drift_success). Nodrift
    agents only count direct docks.

    Args:
        env: The environment instance after episode completion.
        mode: "drift" or "nodrift".

    Returns:
        A tuple of (is_win, outcome_string). outcome_string is one of:
        "direct_dock", "drift_success", "crash", "out_of_bounds",
        "fuel", "timeout", or "unknown".
    """
    # Access the inner environment directly for nodrift (no wrapper layer).
    inner_env = env.env if mode == "drift" else env
    if inner_env.is_docked():
        return True, "direct_dock"

    # Drift agents can also win by reaching a position where coasting
    # to the dock is possible. Nodrift agents must dock directly.
    if mode == "drift":
        is_drift, _ = env.det_drift()
        if is_drift:
            return True, "drift_success"

    reason = classify_failure(env) if mode == "drift" else classify_failure_nodrift(env)
    return False, reason


def find_model_path() -> str:
    """Resolve CHECKPOINT into a single checkpoint file path.

    Returns:
        File path to the one checkpoint this script should evaluate.

    Raises:
        FileNotFoundError: If CHECKPOINT does not resolve to any file.
    """
    checkpoint_files = resolve_checkpoint_paths(
        CHECKPOINT, CHECKPOINT_ROOT, num_models=1, folder_prefix=LATEST_FOLDER_PREFIX
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints resolved for: {CHECKPOINT}")
    return checkpoint_files[-1]


def get_next_figure_number(save_dir: str) -> int:
    """Find the next unused figure number.

    Keeps saved plots from overwriting each other.

    Args:
        save_dir: Folder where trajectory plots are saved.

    Returns:
        The next figure number to use, starting at 1.
    """
    if not os.path.exists(save_dir):
        return 1

    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith("drifter_trajectories_") and f.endswith(".png")
    ]

    numbers = []
    for f in existing:
        try:
            numbers.append(int(f.split("_")[2]))
        except (IndexError, ValueError):
            pass

    return max(numbers) + 1 if numbers else 1


def plot_trajectories(trajectories, episode_outcomes, model_name, model_path, num_episodes,
                      seed=None, show=True):
    """Plot 3D trajectories colored by outcome and save the figure to disk.

    Blue is a win (direct dock or drift success), red is a loss (crash,
    timeout, out of bounds, out of fuel). A single drift stage's
    trajectory is typically very short (5-10 steps); use
    drift_full_test.py for a full long-range approach.

    Args:
        trajectories: One trajectory per episode. Each trajectory is
            a list of (x, y, z) positions over time.
        episode_outcomes: One outcome string per episode, in the same
            order as trajectories.
        model_name: Display name of the model, used in the title.
        model_path: Full path to the model file, used to label which run
            folder this plot came from.
        num_episodes: Total number of episodes run, used in the title.
        seed: The resolved seed, noted in the title so the figure records
            which run produced it.
        show: False saves the PNG without popping up a window (also
            skipped automatically under a headless Matplotlib backend).
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color blind-friendly palette: blue for wins, red for losses.
    WIN_COLOR = "#005AB5"
    LOSS_COLOR = "#DC3220"

    win_plotted = loss_plotted = False

    for trajectory, outcome in zip(trajectories, episode_outcomes):
        traj = np.array(trajectory)

        # Pad single-point trajectories so Matplotlib can draw them.
        if len(traj) < 2:
            traj = np.vstack([traj[0], traj[0] + 1e-6])

        is_win = outcome in ("direct_dock", "drift_success")
        color = WIN_COLOR if is_win else LOSS_COLOR
        label = None
        if is_win and not win_plotted:
            label = "Deputy - Win"
            win_plotted = True
        elif not is_win and not loss_plotted:
            label = "Deputy - Loss"
            loss_plotted = True

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, linewidth=1.5, alpha=0.55, label=label)
        ax.scatter(*traj[0], color=color, s=15, alpha=0.8)  # mark start position

    ax.scatter(0, 0, 0, marker="x", color="black", s=120, linewidths=2.5,
               zorder=5, label="Chief (target)")

    # Force equal and symmetric axis ranges so the chief sits at the center.
    all_points = np.concatenate([np.array(t) for t in trajectories if len(t) > 0])
    set_equal_3d_axes(ax, all_points)
    run_folder = os.path.basename(os.path.dirname(model_path))
    wins_count = sum(1 for o in episode_outcomes if o in ("direct_dock", "drift_success"))
    seed_note = f" (seed {seed})" if seed is not None else ""
    ax.set_title(
        f"Checkpoint Trajectories - {run_folder}/{model_name}{seed_note}\n"
        f"{num_episodes} {plural_word(num_episodes, 'Episode')}, Win rate: {wins_count}/{num_episodes}"
    )
    ax.legend(loc="upper left")
    plt.tight_layout()

    save_dir = os.path.join(SCRIPT_DIR, "saved_figures", "checkpoint_test")
    os.makedirs(save_dir, exist_ok=True)

    figure_num = get_next_figure_number(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir, f"drifter_trajectories_{figure_num:03d}_{timestamp}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print()
    print(f"Saved trajectory plot:\n{save_path}")
    if should_show_plot(show):
        center_matplotlib_window()
        plt.show()
    plt.close()


def evaluate_model(fix_unsafe: bool = True):
    """Run the full evaluation and save a plot and a results file.

    Args:
        fix_unsafe: If True, apply patch_unsafe_termination so episodes
            are not cut short by the training-only speed limit rule.
            Only applies to drift agents. Defaults to True.
    """
    model_path = find_model_path()
    print(f"Loading model:\n{model_path}\n")

    # Resolve the seed before PPO.load(): load() reseeds NumPy's global
    # RNG from the checkpoint's saved seed, so order is significant here.
    resolved_seed = resolve_and_print_seed(SEED)

    # Resolve the agent mode from the config or the checkpoint filename.
    mode = resolve_agent_mode(AGENT_MODE, model_path)
    print(f"Agent mode: {mode}")
    if mode == "drift":
        print(
            "  NOTE: a single drift stage is not a standalone docking "
            "policy; these numbers reflect one stage in isolation. "
            "\n  Use drift_full_test.py (chained) for real drift evaluation."
        )
    print(f"{plural_word(NUM_EPISODES, 'Episode') + ':':<12}{NUM_EPISODES}\n")

    model = PPO.load(model_path, device="cpu")

    # Detect how many observation elements the model expects.
    # Models trained before the 7-element state update expect 6 elements.
    expected_obs_size = model.observation_space.shape[0]
    env, config_source, config_verified = make_env(mode, expected_obs_size, model_path)

    if fix_unsafe and mode == "drift":
        # During evaluation, ignore the training-only unsafe termination
        # rule so checkpoints are judged on docking performance instead.
        patch_unsafe_termination(env)

    # Read positions from the env state (always meters) rather than the
    # observation, which may be normalized (docking_env.py NORMALIZE_OBS).
    inner_env = env.env if mode == "drift" else env

    wins = 0
    direct_docks = 0
    drift_wins = 0
    failure_counts = {
        "crash": 0, "unsafe": 0, "out_of_bounds": 0, "fuel": 0,
        "timeout": 0, "unknown": 0
    }

    fuels = []
    true_fuels = []
    true_fuels_l1 = []
    timesteps = []
    start_dists = []
    end_dists = []
    trajectories = []
    episode_outcomes = []

    # |a| sum = raw action magnitude sum, not Delta-v.
    print(
        f"{'Ep':>4}  {'Start':>8}  {'End':>8}  {'Delta':>8}  "
        f"{'Steps':>6}  {'|a| sum':>9}  {'|a|/step':>10}  Result"
    )
    # Table header separator.
    print("-" * 75)

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset(seed=resolved_seed + episode)
        done = False
        episode_steps = 0
        episode_action_magnitude = 0
        trajectory = [inner_env.state[0:3].copy()]
        start_distance = np.linalg.norm(inner_env.state[0:3])

        while not done:
            # Keep the drift timestep counter within the 5-10 step range
            # the model trained on. Matches drift_full_test.py's limit.
            if mode == "drift" and inner_env.state[6] >= DRIFT_TIMESTEP_CAP:
                inner_env.state[6] = 0

            if mode == "drift" and env.is_drifting:
                # Once det_drift() signals a coast win, hold zero thrust
                # instead of asking the policy again, or fuel use gets
                # overstated.
                action = np.array([0.0, 0.0, 0.0])
            else:
                action, _ = model.predict(obs[:expected_obs_size], deterministic=True)
            # Backwards compatibility: older nodrift models expect 6-element obs, newer ones expect 7 or 9.

            # Suppress environment prints (WIN!, UNSAFE!) so only this
            # script's own per-episode result line is shown.
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append(inner_env.state[0:3].copy())
            # Raw action magnitude, not Delta-v. Divide by 12 for Delta-v.
            episode_action_magnitude += np.linalg.norm(action)
            episode_steps += 1
            done = terminated or truncated
            if episode_steps >= MAX_STEPS_PER_EPISODE:
                break

        end_distance = np.linalg.norm(inner_env.state[0:3])
        if not done:
            # Loop exited via the MAX_STEPS_PER_EPISODE limit, not a valid
            # ending, so classify it as a timeout.
            is_win, outcome = False, "timeout"
        else:
            is_win, outcome = check_win(env, mode)

        if is_win:
            wins += 1
            if outcome == "direct_dock":
                direct_docks += 1
            elif outcome == "drift_success":
                drift_wins += 1
        else:
            failure_counts[outcome] += 1

        distance_delta = end_distance - start_distance
        action_magnitude_per_step = (
            episode_action_magnitude / episode_steps if episode_steps > 0 else 0.0
        )
        print(
            f"{episode+1:>4}  {start_distance:>8.2f}  {end_distance:>8.2f}  {distance_delta:>+8.2f}  "
            f"{episode_steps:>6}  {episode_action_magnitude:>9.3f}  {action_magnitude_per_step:>10.4f}  "
            f"{outcome.upper()}"
        )

        fuels.append(episode_action_magnitude)
        # fuel_used_l1 (per-axis sum) is the primary fuel metric;
        # fuel_used (vector norm) is secondary. See module docstring.
        true_fuels.append(inner_env.fuel_used)
        true_fuels_l1.append(inner_env.fuel_used_l1)
        timesteps.append(episode_steps)
        start_dists.append(start_distance)
        end_dists.append(end_distance)
        trajectories.append(trajectory)
        episode_outcomes.append(outcome)

    env.close()

    print()
    print(f"--- Evaluation Summary ---")
    print(
        f"Model:          "
        f"{os.path.basename(os.path.dirname(model_path))}/"
        f"{os.path.basename(model_path)}"
    )
    print(f"Agent mode:     {mode}")
    print(f"{plural_word(NUM_EPISODES, 'Episode') + ':':<16}{NUM_EPISODES}")
    print(f"Seed:           {resolved_seed}")
    print(f"Win Rate:       {format_win_rate(wins, NUM_EPISODES)}")
    print(f"  Direct docks: {direct_docks}")
    # Only show drift wins line when evaluating a drift agent.
    if mode == "drift":
        print(f"  Drift wins:   {drift_wins}")

    print("\nFailure breakdown:")
    if any(failure_counts.values()):
        for reason, count in failure_counts.items():
            if count:
                print(f"  {reason:<16} {count}")
    else:
        print("  N/A")

    print(
        f"\nTimesteps:  mean={np.mean(timesteps):.1f}  std={np.std(timesteps):.1f}  "
        f"min={np.min(timesteps)}  max={np.max(timesteps)}"
    )
    print(f"  Per episode: {timesteps}")

    # "|a| sum" is the raw action-norm sum, not a fuel metric; the two
    # Delta-v lines below are the real fuel numbers, from fuel_used(_l1).
    print(
        f"Action magnitude (|a| sum, not fuel): mean={np.mean(fuels):.3f}  "
        f"std={np.std(fuels):.3f}  min={np.min(fuels):.3f}  max={np.max(fuels):.3f}"
    )
    # IQM (interquartile mean) reported alongside mean/median to match
    # DRL for Space Controls (2024)'s statistics (see evaluation_utilities). 
    # L1 first: it is the primary fuel metric.
    print(f"Delta-v, per-axis sum (L1, m/s): mean={np.mean(true_fuels_l1):.4f}  "
          f"median={np.median(true_fuels_l1):.4f}  iqm={interquartile_mean(true_fuels_l1):.4f}  "
          f"std={np.std(true_fuels_l1):.4f}")
    print(f"Delta-v, vector norm (L2, m/s):  mean={np.mean(true_fuels):.4f}  "
          f"median={np.median(true_fuels):.4f}  iqm={interquartile_mean(true_fuels):.4f}  "
          f"std={np.std(true_fuels):.4f}")

    print(f"Start dist: mean={np.mean(start_dists):.2f}  std={np.std(start_dists):.2f}")
    print(f"End dist:   mean={np.mean(end_dists):.2f}  std={np.std(end_dists):.2f}")
    print(
        f"Avg delta:  "
        f"{np.mean(np.array(end_dists) - np.array(start_dists)):+.2f}"
    )

    plot_trajectories(
        trajectories, episode_outcomes, os.path.basename(model_path), model_path,
        NUM_EPISODES, seed=resolved_seed, show=SHOW_PLOTS
    )

    results_dir = os.path.join(SCRIPT_DIR, "saved_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"evaluation_{timestamp}.json")

    results = {
        "model": model_path,
        "agent_mode": mode,
        "episodes": NUM_EPISODES,
        "seed": resolved_seed,
        "seed_was_fixed": SEED is not None,
        "env_config_source": config_source,
        "env_config_verified": config_verified,
        "wins": wins,
        "direct_docks": direct_docks,
        "drift_successes": drift_wins,
        "win_rate": wins / NUM_EPISODES,
        "failure_counts": failure_counts,
        "failure_breakdown": "N/A" if not any(failure_counts.values()) else failure_counts,
        "timesteps": {
            "mean": float(np.mean(timesteps)),
            "std": float(np.std(timesteps)),
            "min": int(np.min(timesteps)),
            "max": int(np.max(timesteps)),
            "per_episode": [int(t) for t in timesteps],
        },
        # Raw L2 action-norm sum, not a fuel unit; kept for backward
        # comparison with older logs that used this quantity.
        "action_magnitude": {
            "mean": float(np.mean(fuels)),
            "std": float(np.std(fuels)),
            "min": float(np.min(fuels)),
            "max": float(np.max(fuels)),
        },
        # Delta-v, vector norm: what a single gimbaled thruster would have
        # spent. Secondary metric. Key name kept as "fuel_delta_v_l2_norm"
        # for JSON schema compatibility.
        "fuel_delta_v_l2_norm": {
            "mean": float(np.mean(true_fuels)),
            "median": float(np.median(true_fuels)),
            "iqm": interquartile_mean(true_fuels),
            "std": float(np.std(true_fuels)),
            "min": float(np.min(true_fuels)),
            "max": float(np.max(true_fuels)),
        },
        # Delta-v, per-axis sum: proportional to the propellant the
        # three fixed thrusters use. The primary fuel metric.
        "fuel_delta_v_l1_norm": {
            "mean": float(np.mean(true_fuels_l1)),
            "median": float(np.median(true_fuels_l1)),
            "iqm": interquartile_mean(true_fuels_l1),
            "std": float(np.std(true_fuels_l1)),
            "min": float(np.min(true_fuels_l1)),
            "max": float(np.max(true_fuels_l1)),
        },
        "start_dist": {
            "mean": float(np.mean(start_dists)),
            "std": float(np.std(start_dists)),
        },
        "end_dist": {
            "mean": float(np.mean(end_dists)),
            "std": float(np.std(end_dists)),
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        f.write("\n")

    print()
    print(f"Saved evaluation results:\n{results_path}\n")

    if EXPORT_RESULTS_CSV:
        # One row per episode, long format for Seaborn. Uses
        # write_tidy_csv() since this script's per-episode data
        # doesn't share the paired result dict shape the other scripts use.
        rows = [
            {
                "condition": mode,
                "checkpoint": os.path.basename(model_path),
                "episode_idx": i,
                "outcome": episode_outcomes[i],
                "win": episode_outcomes[i] in ("direct_dock", "drift_success"),
                "fuel_l2": true_fuels[i],
                "fuel_l1": true_fuels_l1[i],
                "episode_length": timesteps[i],
                "start_dist": start_dists[i],
                "end_dist": end_dists[i],
            }
            for i in range(NUM_EPISODES)
        ]
        csv_path = os.path.join(SCRIPT_DIR, "saved_data", f"checkpoint_test_episodes_{timestamp}.csv")
        write_tidy_csv(rows, csv_path, data_label="episode results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Keep the original behavior unless this flag is provided.
    parser.add_argument(
        "--no-fix-unsafe",
        action="store_true",
        help="Disable the patch that removes premature UNSAFE termination.",
    )
    # Override AGENT_MODE from the command line without editing the file.
    parser.add_argument(
        "--mode",
        choices=["drift", "nodrift", "auto"],
        default=None,
        help=(
            "Agent mode: 'drift' for DriftTestEnv, 'nodrift' for "
            "SpaceCraftDockingEnv3D, 'auto' to detect from filename. "
            "Overrides AGENT_MODE set at the top of the file."
        ),
    )
    args = parser.parse_args()

    # Command-line --mode overrides the file-level AGENT_MODE constant.
    if args.mode is not None:
        AGENT_MODE = args.mode

    evaluate_model(fix_unsafe=not args.no_fix_unsafe)
