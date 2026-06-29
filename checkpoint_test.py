"""
checkpoint_test.py

Evaluates a single trained model checkpoint over several episodes and
reports how well it docks: win rate, failure reasons, a 3D trajectory
plot, and a JSON results file.

Checkpoint finding, the early-termination patch, and failure
classification are shared with curriculum_evaluation.py through
evaluation_utilities.py.

Supports both drift-assisted and direct-docking (nodrift) checkpoints.
Set AGENT_MODE below to match the model being evaluated.
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from evaluation_utilities import (
    resolve_checkpoint_paths,
    patch_unsafe_termination,
    classify_failure,
)

# AGENT MODE
# "drift"    -> uses DriftTestEnv from drift_env.py (drift-assisted agent)
# "nodrift"  -> uses SpaceCraftDockingEnv3D from docking_env.py (direct-docking agent)
# "auto"     -> detects from checkpoint filename: "nodrift" in name -> nodrift, else drift
AGENT_MODE = "auto"

# CHECKPOINT CONFIG
# "latest"                          -> newest .zip in the most recent run folder
# "run:safe_PPO_6"                  -> newest .zip inside run folder safe_PPO_6
# "data/checkpoints/safe_PPO_6"     -> newest .zip in this folder
# "pattern:safe_ppo_model_*_6.zip"  -> custom glob inside the checkpoint root
# r"C:\...\checkpoint.zip"          -> exact path to one checkpoint file
CHECKPOINT = "latest"

NUM_EPISODES = 10

# ENV_CONFIG sets the evaluation environment parameters.
# These should match the curriculum stage the model was trained on.
#
# Drift stage 9 example (safe_ppo_model_9_*):
#   pos_thresh = 0.5, speed_thresh = 0.3
#   min/max_init_pos_bound = 100 / 150
#   max_init_vel_bound = 0.5, max_boundary_box = 200
#
# Nodrift stage 4 example (nodrift_ppo_model_4_*):
#   pos_thresh = 10, speed_thresh = 0.22
#   min/max_init_pos_bound = 20 / 75
#   max_init_vel_bound = 0.4, max_boundary_box = 120
#
# Nodrift stage 5 example (nodrift_ppo_model_5_*):
#   pos_thresh = 0.5, speed_thresh = 0.2
#   min/max_init_pos_bound = 75 / 150
#   max_init_vel_bound = 0.5, max_boundary_box = 200
ENV_CONFIG = {
    "min_init_pos_bound": 2.5,
    "max_init_pos_bound": 10,
    "max_init_vel_bound": 0.2,
    "max_boundary_box": 30,
    "max_episode_len": 5,
    "pos_thresh": 2.5,
    "speed_thresh": 0.2,
    "max_total_dv": 1000,
    "drift_step_len": 1,
}

# Folder this script lives in, used to build paths to checkpoints,
# saved figures, and saved results regardless of where the script is
# launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def detect_agent_mode(model_path: str) -> str:
    """Detect whether a checkpoint is a drift or nodrift model.

    Checks the checkpoint filename and parent folder name for the word
    "nodrift". Falls back to "drift" if no match is found.

    Args:
        model_path: Full path to the checkpoint file.

    Returns:
        "nodrift" if the path contains "nodrift", otherwise "drift".
    """
    # Check both the filename and the parent folder name.
    path_lower = model_path.lower()
    if "nodrift" in path_lower:
        return "nodrift"
    return "drift"


def make_env(mode: str):
    """Create the evaluation environment for the given agent mode.

    Args:
        mode: "drift" for DriftTestEnv, "nodrift" for SpaceCraftDockingEnv3D.

    Returns:
        A Gymnasium environment instance configured with ENV_CONFIG.

    Raises:
        ValueError: If mode is not "drift" or "nodrift".
    """
    if mode == "drift":
        from drift_env import DriftTestEnv
        return DriftTestEnv(**ENV_CONFIG)
    elif mode == "nodrift":
        from docking_env import SpaceCraftDockingEnv3D
        return SpaceCraftDockingEnv3D(**ENV_CONFIG)
    else:
        raise ValueError(f"Unknown AGENT_MODE: {mode!r}. Use 'drift', 'nodrift', or 'auto'.")


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
    # Check for a direct dock first, works for both modes.
    inner_env = env.env if mode == "drift" else env
    if inner_env.is_docked():
        return True, "direct_dock"

    # Drift agents can also win by reaching a position where drifting
    # to the dock is possible. Nodrift agents cannot.
    if mode == "drift":
        is_drift, _ = env.det_drift()
        if is_drift:
            return True, "drift_success"

    # If neither condition is met, classify the failure reason.
    reason = classify_failure(env) if mode == "drift" else classify_failure_nodrift(env)
    return False, reason


def classify_failure_nodrift(env) -> str:
    """Classify why a nodrift episode failed.

    Args:
        env: A SpaceCraftDockingEnv3D instance after episode completion.

    Returns:
        A string describing the failure reason.
    """
    if env.is_crashed():
        return "crash"
    if env.is_out_of_bounds():
        return "out_of_bounds"
    if env.is_out_of_fuel():
        return "fuel"
    if env.is_out_of_time():
        return "timeout"
    return "unknown"


def find_model_path() -> str:
    """Resolve CHECKPOINT into a single checkpoint file path.

    Returns:
        File path to the one checkpoint this script should evaluate.

    Raises:
        FileNotFoundError: If CHECKPOINT does not resolve to any file.
    """
    checkpoint_files = resolve_checkpoint_paths(
        CHECKPOINT, CHECKPOINT_ROOT, num_models=1
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints resolved for: {CHECKPOINT}")
    return checkpoint_files[-1]


def get_next_figure_number(save_dir: str) -> int:
    """Find the next unused figure number so saved plots do not overwrite
    each other.

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


def plot_trajectories(all_trajectories, outcomes, model_name, model_path_global):
    """Plot 3D trajectories colored by outcome and save the figure to disk.

    Blue trajectories are wins (direct dock or drift success), red
    trajectories are losses (crash, timeout, out of bounds, out of fuel).

    Args:
        all_trajectories: One trajectory per episode. Each trajectory is
            a list of (x, y, z) positions over time.
        outcomes: One outcome string per episode, in the same order as
            all_trajectories.
        model_name: Display name of the model, used in the title.
        model_path_global: Full path to the model file, used to label
            which run folder this plot came from.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color blind-friendly palette: blue for wins, red for losses.
    WIN_COLOR = "#005AB5"
    LOSS_COLOR = "#DC3220"

    win_plotted = loss_plotted = False

    for trajectory, outcome in zip(all_trajectories, outcomes):
        traj = np.array(trajectory)

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
        ax.scatter(*traj[0], color=color, s=15, alpha=0.8)

    ax.scatter(0, 0, 0, marker="x", color="black", s=120, linewidths=2.5,
               zorder=5, label="Chief (target)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    run_folder = os.path.basename(os.path.dirname(model_path_global))
    ax.set_title(
        f"Drifter Trajectories - {len(all_trajectories)} Episodes\n"
        f"{run_folder}/{model_name}"
    )
    ax.legend(loc="upper left")
    plt.tight_layout()

    save_dir = os.path.join(SCRIPT_DIR, "saved_figures")
    os.makedirs(save_dir, exist_ok=True)

    figure_num = get_next_figure_number(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir, f"drifter_trajectories_{figure_num:03d}_{timestamp}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved trajectory plot:\n{save_path}")
    plt.show()


def evaluate_model(fix_unsafe: bool = True):
    """Run the full evaluation and save a plot and a results file.

    Args:
        fix_unsafe: If True, apply patch_unsafe_termination so episodes
            are not cut short by the training-only speed limit rule.
            Only applies to drift agents. Defaults to True.
    """
    model_path = find_model_path()
    print(f"Loading model:\n{model_path}\n")

    # Resolve the agent mode from the config or the checkpoint filename.
    mode = AGENT_MODE
    if mode == "auto":
        mode = detect_agent_mode(model_path)
    print(f"Agent mode: {mode}\n")

    model = PPO.load(model_path, device="cpu")
    env = make_env(mode)

    # Detect how many observation elements the model expects.
    # Models trained before the 7-element state update expect 6 elements.
    expected_obs_size = model.observation_space.shape[0]

    # The unsafe termination patch only applies to drift agents.
    # Nodrift agents do not use the is_unsafe() termination condition.
    if fix_unsafe and mode == "drift":
        patch_unsafe_termination(env)

    wins = 0
    direct_docks = 0
    drift_wins = 0

    failure_counts = {
        "crash": 0, "out_of_bounds": 0, "fuel": 0, "timeout": 0, "unknown": 0
    }

    fuels = []
    timesteps = []
    start_dists = []
    end_dists = []
    all_trajectories = []
    outcomes = []

    print(
        f"{'Ep':>4}  {'Start':>8}  {'End':>8}  {'Delta':>8}  "
        f"{'Steps':>6}  {'Fuel':>8}  {'Fuel/Step':>10}  Result"
    )
    print("-" * 75)

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()

        done = False
        episode_steps = 0
        episode_fuel = 0
        trajectory = [obs[:3].copy()]

        start_dist = np.linalg.norm(obs[:3])

        while not done:
            action, _ = model.predict(obs[:expected_obs_size], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append(obs[:3].copy())
            episode_fuel += np.linalg.norm(action)
            episode_steps += 1
            done = terminated or truncated

        end_dist = np.linalg.norm(obs[:3])

        is_win, outcome = check_win(env, mode)
        if is_win:
            wins += 1
            if outcome == "direct_dock":
                direct_docks += 1
            elif outcome == "drift_success":
                drift_wins += 1
        else:
            failure_counts[outcome] += 1

        delta = end_dist - start_dist
        fuel_per_step = episode_fuel / episode_steps if episode_steps > 0 else 0.0
        print(
            f"{episode+1:>4}  {start_dist:>8.2f}  {end_dist:>8.2f}  {delta:>+8.2f}  "
            f"{episode_steps:>6}  {episode_fuel:>8.3f}  {fuel_per_step:>8.4f}  "
            f"{outcome.upper()}"
        )

        fuels.append(episode_fuel)
        timesteps.append(episode_steps)
        start_dists.append(start_dist)
        end_dists.append(end_dist)
        all_trajectories.append(trajectory)
        outcomes.append(outcome)

    env.close()

    print("\n" + "=" * 65)
    print("Evaluation Summary")
    print("=" * 65)
    print(
        f"Model:          "
        f"{os.path.basename(os.path.dirname(model_path))}/"
        f"{os.path.basename(model_path)}"
    )
    print(f"Agent mode:     {mode}")
    print(f"Episodes:       {NUM_EPISODES}")
    print(f"Win Rate:       {wins}/{NUM_EPISODES}  ({100*wins/NUM_EPISODES:.1f}%)")
    print(f"  Direct docks: {direct_docks}")
    # Only show drift wins line when evaluating a drift agent.
    if mode == "drift":
        print(f"  Drift wins:   {drift_wins}")

    print("\nFailure breakdown:")
    for reason, count in failure_counts.items():
        if count:
            print(f"  {reason:<16} {count}")

    print(
        f"\nTimesteps:  mean={np.mean(timesteps):.1f}  std={np.std(timesteps):.1f}  "
        f"min={np.min(timesteps)}  max={np.max(timesteps)}"
    )
    print(f"  Per episode: {timesteps}")
    print(
        f"Fuel:       mean={np.mean(fuels):.3f}  std={np.std(fuels):.3f}  "
        f"min={np.min(fuels):.3f}  max={np.max(fuels):.3f}"
    )
    print(f"Start dist: mean={np.mean(start_dists):.2f}  std={np.std(start_dists):.2f}")
    print(f"End dist:   mean={np.mean(end_dists):.2f}  std={np.std(end_dists):.2f}")
    print(
        f"Avg delta:  "
        f"{np.mean(np.array(end_dists) - np.array(start_dists)):+.2f}"
    )

    plot_trajectories(
        all_trajectories, outcomes, os.path.basename(model_path), model_path
    )

    results_dir = os.path.join(SCRIPT_DIR, "saved_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"evaluation_{timestamp}.json")

    results = {
        "model": model_path,
        "agent_mode": mode,
        "episodes": NUM_EPISODES,
        "env_config": ENV_CONFIG,
        "wins": wins,
        "direct_docks": direct_docks,
        "drift_successes": drift_wins,
        "win_rate": wins / NUM_EPISODES,
        "failure_counts": failure_counts,
        "timesteps": {
            "mean": float(np.mean(timesteps)),
            "std": float(np.std(timesteps)),
            "min": int(np.min(timesteps)),
            "max": int(np.max(timesteps)),
            "per_episode": [int(t) for t in timesteps],
        },
        "fuel": {
            "mean": float(np.mean(fuels)),
            "std": float(np.std(fuels)),
            "min": float(np.min(fuels)),
            "max": float(np.max(fuels)),
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

    print(f"\nSaved evaluation results:\n{results_path}")


# Entry point for running this file directly.
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
