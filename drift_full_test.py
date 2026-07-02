"""
drift_full_test.py

Tests a chain of curriculum models end-to-end in a single continuous episode.

Starting from the hardest model (model 5, farthest starting range), the script
runs the agent until that stage's docking condition is met, then switches to
the next easier model and continues in the same environment. Only the final
stage's result counts as a win or loss for the episode.

Related evaluation scripts:
    - drift_full_test.py (this script):
        End-to-end drift-assisted curriculum chain in a single continuous
        episode.
        Use when: Evaluating the full mission flow of the drift agent.
    
    - checkpoint_test.py:
        Evaluates one checkpoint on one curriculum stage over multiple
        episodes.
        Use when: Diagnosing a single model in its own training setup.

    - curriculum_evaluation.py:
        Automatically resolves a checkpoint set and evaluates the full
        curriculum chain with a fixed seed.
        Use when: Repeatable, headless results are needed from a
        checkpoint folder.

    - nodrift_full_test.py:
        End-to-end direct-docking curriculum chain in a single continuous
        episode.
        Use when: Comparing direct-docking performance against the drift
        agent.

Differences from curriculum_evaluation.py:
    - Manual setup: model paths and thresholds set directly at the top.
    - No fixed random seed: each run has different starting conditions.
    - Single episode flow: switches models mid-episode until final stage.

The environment itself prints "WIN!" when is_docked() is true. This is
suppressed via contextlib.redirect_stdout so only this script's own
"Model N: WIN/FAIL" line is shown.

Update the model paths at the top of the script to match your run folder
before running.
"""

import copy
import contextlib
import io
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from drift_env import DriftTestEnv
from drift_initial_trainer import get_curriculum
from stable_baselines3 import PPO

# Build the checkpoint path relative to this file so the script works
# regardless of which directory it is launched from.
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"

# Load each curriculum stage model. Update these paths to match your run folder.
# Each model was trained on a progressively harder range of starting distances.
# Format: safe_ppo_model_{run}_{stage}_{epoch}.zip
# Example: safe_ppo_model_14_6_9.zip -> run 14, stage 6, epoch 9
model_1 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_14" /
                       "safe_ppo_model_14_1_0.zip"))
model_2 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_14" /
                       "safe_ppo_model_14_3_1.zip"))
model_3 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_14" /
                       "safe_ppo_model_14_6_9.zip"))
model_4 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_14" /
                       "safe_ppo_model_14_7_5.zip"))
model_5_path = CHECKPOINT_DIR / "safe_PPO_14" / \
                       "safe_ppo_model_14_8_2.zip"
model_5 = PPO.load(str(model_5_path))

# Name of the run folder model_5 was loaded from, used to label the
# trajectory plot. Pulled from model_5's path rather than hardcoded, since
# that is usually the hardest/final stage and the most representative model
# for this run. 
# Note: if the five models above are loaded from different run folders, 
# this will only reflect model_5's folder.
RUN_FOLDER_NAME = model_5_path.parent.name

# Number of episodes to run.
NUM_EPISODES = 10

# True: print the episode index, env config, and per-step debug info.
# False: only print model-switch progress and the final summary.
VERBOSE = False

# Ablation flags. True: enable, False: use default behavior.
# USE_SAFE_ACTION: replace the model's raw action with the result of a
# one-step lookahead safety check (see get_safe_action below).
USE_SAFE_ACTION = False
# USE_ACTION_NOISE: multiply every action by random noise in [0.95, 1.05]
# to test robustness to actuator or sensor uncertainty.
USE_ACTION_NOISE = False

# True: save a 3D trajectory plot of NUM_PLOT_EPISODES episodes to saved_figures/.
# False: skip plotting entirely.
PLOT_TRAJECTORIES = True
# How many of the first episodes to record and plot. Defaults to NUM_EPISODES
# so every episode is plotted. Set to a smaller number (e.g. 10) if
# NUM_EPISODES is large and plotting all of them would be slow or cluttered.
NUM_PLOT_EPISODES = NUM_EPISODES


def get_safe_action(env, action, obs):
    """Pick a safe action, falling back to zero thrust or a braking action.

    This is an ablation for testing whether a one-step lookahead safety
    check improves docking outcomes. Tries the model's chosen action first,
    then zero thrust, then thrust opposite to the current velocity, and
    returns the first one that does not trigger is_unsafe() in a copy of
    the environment.

    Only called when USE_SAFE_ACTION is True.

    Args:
        env: The current DriftTestEnv instance.
        action: The action the model predicted for this step.
        obs: The current observation, used to reset the state of a
            temporary environment copy before testing each candidate action.

    Returns:
        The first candidate action found to be safe, or the original
        action if none of the candidates are safe.
    """
    action_list = []
    action_list.append(action)
    action_list.append(np.array([0.0, 0.0, 0.0]))
    action_list.append(-1 * env.env.state[3:6])

    for tmp_action in action_list:
        tmp_env = copy.deepcopy(env)
        tmp_env.env.state = obs
        tmp_env.step(tmp_action)
        if not tmp_env.env.is_unsafe():
            return tmp_action

    print('NO SAFE ACTION')
    return action


def plot_full_trajectories(trajectories, outcomes, num_episodes, run_folder):
    """Plot 3D trajectories of full mission chain episodes and save to disk.

    Each trajectory covers the full chain from the farthest starting range
    down to the final precise dock. Blue trajectories are wins, red are losses.

    Args:
        trajectories: List of trajectories, one per episode. Each trajectory
            is a list of (x, y, z) positions recorded every step.
        outcomes: List of outcome strings, one per episode.
        num_episodes: Total number of episodes run, used in the title.
        run_folder: Name of the checkpoint run folder (e.g. "safe_PPO_14"),
            shown in the title so the plot can be traced back to its models.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    WIN_COLOR = "#005AB5"
    LOSS_COLOR = "#DC3220"

    win_plotted = loss_plotted = False

    for traj, outcome in zip(trajectories, outcomes):
        traj = np.array(traj)
        if len(traj) < 2:
            continue

        is_win = outcome == "win"
        color = WIN_COLOR if is_win else LOSS_COLOR
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

    # Chief spacecraft is always at the origin.
    ax.scatter(0, 0, 0, marker="x", color="black", s=150, linewidths=2.5,
               zorder=5, label="Chief (target)")

    # Force equal, symmetric axis ranges so the chief sits at the center
    # of the plot rather than off to one side.
    all_points = np.concatenate([np.array(t) for t in trajectories if len(t) > 0])
    max_extent = np.max(np.abs(all_points))
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Full Mission Chain Trajectories - {run_folder}\n"
                 f"{num_episodes} Episodes, Win rate: "
                 f"{sum(1 for o in outcomes if o == 'win')}/{num_episodes}")
    ax.legend(loc="upper left")
    plt.tight_layout()

    save_dir = BASE_DIR / "saved_figures"
    save_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"full_chain_trajectories_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved trajectory plot:\n{save_path}")
    plt.show()


fuels = []      # total fuel used per episode
wins = []       # 1 for a successful dock, 0 for a failure
all_trajectories = []   # full position trajectory per episode, if plotting
all_outcomes = []       # win or loss per episode, if plotting

for i in range(NUM_EPISODES):
    print(f'Episode {i + 1}/{NUM_EPISODES}')
    if VERBOSE:
        print('starting at model 5')
    epi_fuel = 0

    # Create a fresh curriculum and environment for each episode.
    # get_curriculum(10) returns the test environment config.
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)
    obs, info = env.reset()
    done = False

    # Start each episode with the hardest model (model 5, farthest range).
    # The script steps down to easier models as each stage completes.
    model_num = 5
    model = model_5

    # env.env.state[3] = 0.0
    # env.env.state[4] = 0.0
    # env.env.state[5] = 0.0

    # Record the trajectory for this episode if plotting is enabled.
    episode_trajectory = [obs[:3].copy()] if PLOT_TRAJECTORIES and i < NUM_PLOT_EPISODES else None

    while not done:
        # obs[:-1] *= np.random.uniform(0.99, 1.01, 1)

        # Keep the timestep counter within the range seen during training.
        # Each stage is generally trained with a small max_episode_len (6-10 steps),
        # so the model may behave unexpectedly if state[6] goes beyond that.
        if env.env.state[6] >= 10:
            env.env.state[6] = 0

        if env.is_drifting:
            # During a drift period the agent holds position with zero thrust.
            action = np.array([0.0, 0.0, 0.0])
        else:
            # Slice obs to match the observation size the model was trained on.
            # Older models expect 6 elements; newer ones expect 7 (with timestep).
            expected_obs_size = model.observation_space.shape[0]
            action = model.predict(obs[:expected_obs_size], deterministic=True)[0]
            if USE_SAFE_ACTION:
                action = get_safe_action(env, action, obs)

        if USE_ACTION_NOISE:
            action = action * np.random.uniform(0.95, 1.05)

        # Accumulate fuel as the sum of absolute thrust across all axes,
        # divided by mass and multiplied by step length to get delta-V units.
        epi_fuel += float(np.sum(np.abs(action))) / env.env.m * env.env.step_len

        # Stdout is suppressed here because the environment itself prints
        # "WIN!" on a successful dock, which would duplicate the
        # "Model N: WIN/FAIL" line printed below.
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, term, trunc, info = env.step(action)
        if VERBOSE:
            print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        done = term or trunc

        if episode_trajectory is not None:
            episode_trajectory.append(obs[:3].copy())

        if done:
            stage_result = 'WIN' if env.env.is_docked() else 'FAIL'
            print(f'  Model {model_num}: {stage_result}')

            if model_num == 1:
                # The easiest model just finished, so this is the final result
                # for the whole episode. Record win or loss and total fuel used.
                result = env.env.is_docked()
                wins.append(1) if result else wins.append(0)
                fuels.append(epi_fuel)
                if episode_trajectory is not None:
                    all_trajectories.append(episode_trajectory)
                    all_outcomes.append("win" if result else "loss")
            else:
                # This stage finished but the chain is not done yet.
                # Reset the episode-ending flags and switch to the next easier
                # model, then keep stepping in the same environment.
                done = False
                env.is_drifting = False
                env.env.state[6] = 0  # reset the timestep counter in the state vector
                model_num -= 1

                # env.env.state[3] = 0.0
                # env.env.state[4] = 0.0
                # env.env.state[5] = 0.0

                # Update the docking thresholds to match each model's training config.
                # Each stage was trained with a different position and speed target.
                if model_num == 4:
                    model = model_4
                    env.env.dock_dist = 50
                    env.env.dock_speed = 0.30
                elif model_num == 3:
                    model = model_3
                    env.env.dock_dist = 10
                    env.env.dock_speed = 0.22
                elif model_num == 2:
                    model = model_2
                    env.env.dock_dist = 2.5
                    env.env.dock_speed = 0.2
                else:
                    model = model_1
                    env.env.dock_dist = 0.5
                    env.env.dock_speed = 0.2

    print()  # blank line between episodes

# Print a summary of results across all episodes.
print(f'Results over {len(wins)} episodes:')
print(f'  Win rate:  {sum(wins)}/{len(wins)} ({100 * np.mean(wins):.1f}%)')

print(f'\nFuel (delta-V):')
print(f'  Mean:      {np.mean(fuels):.3f}')
print(f'  Median:    {np.median(fuels):.3f}')
print(f'  Min:       {min(fuels):.3f}')
print(f'  Max:       {max(fuels):.3f}')
print(f'  25th pct:  {np.percentile(fuels, 25):.3f}')
print(f'  50th pct:  {np.percentile(fuels, 50):.3f}')
print(f'  75th pct:  {np.percentile(fuels, 75):.3f}')

# Plot trajectories if enabled and we have data.
if PLOT_TRAJECTORIES and all_trajectories:
    plot_full_trajectories(all_trajectories, all_outcomes, len(wins), RUN_FOLDER_NAME)
