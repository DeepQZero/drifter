"""
checkpoint_test.py

Evaluates a single trained model checkpoint over several episodes and
reports how well it docks: win rate, failure reasons, a 3D trajectory
plot, and a JSON results file.

Checkpoint finding, the early-termination patch, and failure
classification are shared with curriculum_evaluation.py through
evaluation_utilities.py.
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from drift_env import DriftTestEnv
from evaluation_utilities import (
    resolve_checkpoint_paths,
    patch_unsafe_termination,
    classify_failure,
)

# CHECKPOINT CONFIG
# "latest"                          -> newest .zip in the most recent run folder
# "run:safe_PPO_6"                  -> newest .zip inside run folder safe_PPO_6
# "data/checkpoints/safe_PPO_6"     -> newest .zip in this folder
# "pattern:safe_ppo_model_*_6.zip"  -> custom glob inside the checkpoint root
# r"C:\...\checkpoint.zip"          -> exact path to one checkpoint file
CHECKPOINT = "latest"

NUM_EPISODES = 10

# Generally matches the curriculum stage used for training.

# This matches get_curriculum(9):
#   pos_thresh = 0.5
#   speed_thresh = 0.3
#   min/max_init_pos_bound = 100 / 150
#   max_init_vel_bound = 0.5
#   max_boundary_box = 200
ENV_CONFIG = {
    "min_init_pos_bound": 100,
    "max_init_pos_bound": 150,
    "max_init_vel_bound": 0.5,
    "max_boundary_box": 200,
    "max_episode_len": 1500,
    "pos_thresh": 0.5,
    "speed_thresh": 0.3,
}

# Folder this script lives in, used to build paths to checkpoints,
# saved figures, and saved results regardless of where the script is
# launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def find_model_path():
    """Resolve CHECKPOINT into a single checkpoint file path.

    Returns:
        str: File path to the one checkpoint this script should evaluate.

    Raises:
        FileNotFoundError: If CHECKPOINT does not resolve to any file.
    """
    checkpoint_files = resolve_checkpoint_paths(
        CHECKPOINT, CHECKPOINT_ROOT, num_models=1
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints resolved for: {CHECKPOINT}")
    return checkpoint_files[-1]


# Trajectory plotting
def get_next_figure_number(save_dir):
    """Find the next unused figure number so saved plots do not overwrite
    each other.

    Args:
        save_dir (str): Folder where trajectory plots are saved.

    Returns:
        int: The next figure number to use, starting at 1.
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
        all_trajectories (list): One trajectory per episode. Each
            trajectory is a list of (x, y, z) positions over time.
        outcomes (list): One outcome string per episode, in the same
            order as all_trajectories.
        model_name (str): Display name of the model, used in the title.
        model_path_global (str): Full path to the model file, used to
            label which run folder this plot came from.
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
    ax.set_title(f"Drifter Trajectories - {len(all_trajectories)} Episodes\n{run_folder}/{model_name}")
    ax.legend(loc="upper left")
    plt.tight_layout()

    save_dir = os.path.join(SCRIPT_DIR, "saved_figures")
    os.makedirs(save_dir, exist_ok=True)

    figure_num = get_next_figure_number(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"drifter_trajectories_{figure_num:03d}_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved trajectory plot:\n{save_path}")
    plt.show()


# Main evaluation loop
def evaluate_model(fix_unsafe: bool = True):
    """Run the full evaluation and save a plot and a results file.

    Args:
        fix_unsafe (bool): If True, apply patch_unsafe_termination so
            episodes are not cut short by the training-only speed limit
            rule. Defaults to True.
    """
    model_path = find_model_path()
    print(f"Loading model:\n{model_path}\n")

    model = PPO.load(model_path, device="cpu")

    env = DriftTestEnv(**ENV_CONFIG)

    if fix_unsafe:
        patch_unsafe_termination(env)

    wins = 0
    direct_docks = 0
    drift_wins = 0

    failure_counts = {"crash": 0, "out_of_bounds": 0, "fuel": 0, "timeout": 0, "unknown": 0}

    fuels = []
    timesteps = []
    start_dists = []
    end_dists = []

    all_trajectories = []
    outcomes = []

    print(f"{'Ep':>4}  {'Start':>8}  {'End':>8}  {'Delta':>8}  {'Steps':>6}  {'Fuel':>8}  {'Fuel/Step':>10}  Result")
    print("-" * 75)

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()

        done = False
        episode_steps = 0
        episode_fuel = 0
        trajectory = [obs[:3].copy()]

        start_dist = np.linalg.norm(obs[:3])

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append(obs[:3].copy())
            episode_fuel += np.linalg.norm(action)
            episode_steps += 1
            done = terminated or truncated

        end_dist = np.linalg.norm(obs[:3])

        if env.env.is_docked():
            outcome = "direct_dock"
            wins += 1
            direct_docks += 1
        else:
            is_drift, _ = env.det_drift()
            if is_drift:
                outcome = "drift_success"
                wins += 1
                drift_wins += 1
            else:
                reason = classify_failure(env)
                outcome = reason
                failure_counts[reason] += 1

        delta = end_dist - start_dist
        fuel_per_step = episode_fuel / episode_steps if episode_steps > 0 else 0.0
        print(f"{episode+1:>4}  {start_dist:>8.2f}  {end_dist:>8.2f}  {delta:>+8.2f}  "
              f"{episode_steps:>6}  {episode_fuel:>8.3f}  {fuel_per_step:>8.4f}  {outcome.upper()}")

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
    print(f"Model:          {os.path.basename(os.path.dirname(model_path))}/{os.path.basename(model_path)}")
    print(f"Episodes:       {NUM_EPISODES}")
    print(f"Win Rate:       {wins}/{NUM_EPISODES}  ({100*wins/NUM_EPISODES:.1f}%)")
    print(f"  Direct docks: {direct_docks}")
    print(f"  Drift wins:   {drift_wins}")

    print("\nFailure breakdown:")
    for reason, count in failure_counts.items():
        if count:
            print(f"  {reason:<16} {count}")

    print(f"\nTimesteps:  mean={np.mean(timesteps):.1f}  std={np.std(timesteps):.1f}  "
          f"min={np.min(timesteps)}  max={np.max(timesteps)}")
    print(f"  Per episode: {timesteps}")
    print(f"Fuel:       mean={np.mean(fuels):.3f}  std={np.std(fuels):.3f}  "
          f"min={np.min(fuels):.3f}  max={np.max(fuels):.3f}")
    print(f"Start dist: mean={np.mean(start_dists):.2f}  std={np.std(start_dists):.2f}")
    print(f"End dist:   mean={np.mean(end_dists):.2f}  std={np.std(end_dists):.2f}")
    print(f"Avg delta:  {np.mean(np.array(end_dists) - np.array(start_dists)):+.2f}")

    plot_trajectories(all_trajectories, outcomes, os.path.basename(model_path), model_path)

    results_dir = os.path.join(SCRIPT_DIR, "saved_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"evaluation_{timestamp}.json")

    results = {
        "model": model_path,
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
    # Run evaluation with the unsafe-termination fix enabled by default.
    args = parser.parse_args()
    evaluate_model(fix_unsafe=not args.no_fix_unsafe)
