"""
drift_full_test.py

Tests a chain of curriculum models end to end in a single continuous
episode. Starts on the hardest model (model 5, farthest starting range),
runs until that stage docks, then switches to the next easier model in
the same environment. Only the final stage's result counts as a win or
loss.

Related evaluation scripts:
    - checkpoint_test.py: one checkpoint, one stage, several episodes.
      Use to diagnose a single model in its own training setup.
    - curriculum_evaluation.py: auto-resolves a checkpoint set and chains
      the full curriculum with a fixed seed. Use for repeatable results 
      from a checkpoint folder.
    - nodrift_full_test.py: the same idea, for the direct-docking agent.
      Use to compare against the drift agent.

Differences from curriculum_evaluation.py:
    - Model paths and thresholds are set manually at the top of the file.
    - No fixed seed, so each run gets different starting conditions.

The environment prints "WIN!" on a successful dock. This script
suppresses that via contextlib.redirect_stdout so only its own
"Model N: WIN/FAIL" line shows.

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
# Example: safe_ppo_model_15_6_9.zip -> run 15, stage 6, epoch 9
model_1 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_16" /
                       "safe_ppo_model_16_1_0.zip"))
model_2 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_16" /
                       "safe_ppo_model_16_3_3.zip"))
model_3 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_16" /
                       "safe_ppo_model_16_6_9.zip"))
model_4 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_16" /
                       "safe_ppo_model_16_7_0.zip"))
model_5_path = CHECKPOINT_DIR / "safe_PPO_16" / \
                       "safe_ppo_model_16_8_3.zip"
model_5 = PPO.load(str(model_5_path))

# A 9-element observation means the models were trained with SafeRL
# observations on (see drift_env.py's SAFERL_OBS flag); build the
# environment to match so the observation layout lines up.
SAFERL_OBS = model_5.observation_space.shape[0] == 9

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
# USE_ACTION_NOISE: If True, multiply actions by random noise in [0.95, 1.05].
# If False, use the model's raw action.
USE_ACTION_NOISE = False

# True: save a 3D trajectory plot of NUM_PLOT_EPISODES episodes to saved_figures/.
# False: skip plotting entirely.
PLOT_TRAJECTORIES = True
# How many of the first episodes to record and plot. Defaults to NUM_EPISODES
# so every episode is plotted. Set to a smaller number (e.g. 10) if
# NUM_EPISODES is large and plotting all of them would be slow or cluttered.
NUM_PLOT_EPISODES = NUM_EPISODES

# True: print the thrust vector, its magnitude, and current position/speed
# every PRINT_THRUST_EVERY steps, so you can watch what the agent is doing
# as it flies (including drift periods, where thrust reads as zero).
# False: no per-step thrust output.
PRINT_THRUST = False
PRINT_THRUST_EVERY = 10  # print every Nth step; 1 = every step

# True: color each plotted trajectory by thrust magnitude at each point
# (bright = hard burn, dark = coasting/low thrust) instead of a flat
# win/loss color. Useful here to see where the drift periods actually are.
# False: use the plain win/loss coloring.
COLOR_TRAJECTORY_BY_THRUST = True

# Colormap used when COLOR_TRAJECTORY_BY_THRUST is enabled.
# Options:
#   - "grey_to_warm": default, grey to orange/red gradient
#      grey = drifting, orange/red = hard burn
#   - "viridis": uniform, colorblind-safe gradient
THRUST_COLORMAP = "grey_to_warm"


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


def plot_full_trajectories(trajectories, thrusts, outcomes, num_episodes, run_folder):
    """Plot 3D trajectories of full mission chain episodes and save to disk.

    Each trajectory covers the full chain from the farthest starting range
    down to the final precise dock.

    If COLOR_TRAJECTORY_BY_THRUST is True, each trajectory segment is colored
    by how hard the agent was thrusting at that point (a colorblind-friendly
    colormap from low to high thrust magnitude), so drift/coast periods show
    up as dark stretches and burns show up bright. Otherwise, trajectories
    are colored flat blue (win) or red (loss).

    Args:
        trajectories: List of trajectories, one per episode. Each trajectory
            is a list of (x, y, z) positions recorded every step.
        thrusts: List of per-step thrust magnitudes, parallel to
            trajectories (thrusts[i][j] corresponds to trajectories[i][j]).
        outcomes: List of outcome strings, one per episode.
        num_episodes: Total number of episodes run, used in the title.
        run_folder: Name of the checkpoint run folder (e.g. "safe_PPO_16"),
            shown in the title so the plot can be traced back to its models.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Color blind-friendly palette
    WIN_COLOR = "#005AB5"
    LOSS_COLOR = "#DC3220"

    win_plotted = loss_plotted = False
    thrust_line = None  # last Line3DCollection drawn, used to anchor the colorbar

    if COLOR_TRAJECTORY_BY_THRUST:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import LinearSegmentedColormap
        # Shared color scale across all episodes so brightness is comparable
        # between trajectories, not just within one.
        max_thrust = max((max(t) for t in thrusts if len(t) > 0), default=1.0) or 1.0
        if THRUST_COLORMAP == "grey_to_warm":
            thrust_cmap = LinearSegmentedColormap.from_list(
                "grey_to_warm",
                ["#969393", "#FFDD46", "#E5402E"],
            )
        elif THRUST_COLORMAP == "viridis":
            thrust_cmap = "viridis"
        else:
            raise ValueError(
                f"Unsupported THRUST_COLORMAP: {THRUST_COLORMAP!r}. "
                'Use "grey_to_warm" or "viridis".'
            )

    for traj, thrust, outcome in zip(trajectories, thrusts, outcomes):
        traj = np.array(traj)
        if len(traj) < 2:
            continue

        is_win = outcome == "win"
        color = WIN_COLOR if is_win else LOSS_COLOR

        if COLOR_TRAJECTORY_BY_THRUST:
            # Build one small line segment per step and color it by the
            # thrust magnitude used on that step (bright = hard burn).
            points = traj.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_thrust = np.array(thrust[1:len(traj)])
            if THRUST_COLORMAP == "viridis":
                # viridis: perceptually uniform and colorblind-safe
                lc = Line3DCollection(segments, cmap="viridis", alpha=0.8, linewidth=1.5)
            else:
                lc = Line3DCollection(segments, cmap=thrust_cmap, alpha=0.8, linewidth=1.5)
            lc.set_array(segment_thrust)
            lc.set_clim(0, max_thrust)
            ax.add_collection3d(lc)
            thrust_line = lc
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
        cbar = fig.colorbar(thrust_line, ax=ax, shrink=0.6, pad=0.04)
        cbar.set_label("Thrust magnitude (sum |thrust| per step)", labelpad=10)

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
    ax.set_title(f"Drifter Full Mission Chain Trajectories - {run_folder}\n"
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
all_thrusts = []        # per-step thrust magnitude, parallel to all_trajectories
all_outcomes = []       # win or loss per episode, if plotting

for i in range(NUM_EPISODES):
    print(f'Episode {i + 1}/{NUM_EPISODES}')
    if VERBOSE:
        print('starting at model 5')
    epi_fuel = 0

    # Create a fresh curriculum and environment for each episode.
    # get_curriculum(10) returns the test environment config.
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(saferl_obs=SAFERL_OBS, **curriculum)
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
    episode_thrusts = [0.0] if episode_trajectory is not None else None

    step_count = 0
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

        thrust_mag = float(np.sum(np.abs(action)))

        # Accumulate fuel as the sum of absolute thrust across all axes,
        # divided by mass and multiplied by step length to get delta-V units.
        epi_fuel += thrust_mag / env.env.m * env.env.step_len

        # Stdout is suppressed here because the environment itself prints
        # "WIN!" on a successful dock, which would duplicate the
        # "Model N: WIN/FAIL" line printed below.
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, term, trunc, info = env.step(action)
        if VERBOSE:
            print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        if PRINT_THRUST and step_count % PRINT_THRUST_EVERY == 0:
            pos_norm = np.linalg.norm(obs[0:3])
            speed_norm = np.linalg.norm(obs[3:6])
            print(f'    t={step_count:4d}  thrust={np.round(action, 3)}  '
                  f'|thrust|={thrust_mag:.3f}  dist={pos_norm:.2f}m  speed={speed_norm:.3f}m/s'
                  f'{"  DRIFTING" if env.is_drifting else "  THRUSTING"}')
        done = term or trunc
        step_count += 1

        if episode_trajectory is not None:
            episode_trajectory.append(obs[:3].copy())
            episode_thrusts.append(thrust_mag)

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
                    all_thrusts.append(episode_thrusts)
                    all_outcomes.append("win" if result else "loss")
            else:
                # This stage finished but the chain is not done yet.
                # Reset the episode-ending flags and switch to the next easier
                # model, then keep stepping in the same environment.
                done = False
                env.is_drifting = False
                env.env.state[6] = 0  # reset the timestep counter in the state vector
                env.env.fuel_used = 0 # reset fuel counter so the next stage isn't
                                      # charged for fuel spent on an earlier stage
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

# Plot trajectories (if enabled)
if PLOT_TRAJECTORIES and all_trajectories:
    plot_full_trajectories(all_trajectories, all_thrusts, all_outcomes, len(wins), RUN_FOLDER_NAME)
