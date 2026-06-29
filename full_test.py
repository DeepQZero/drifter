"""
full_test.py

Tests a chain of curriculum models end-to-end in a single continuous episode.

Starting from the hardest model (model 5, farthest starting range), the script
runs the agent until that stage's docking condition is met, then switches to
the next easier model and continues in the same environment. Only the final
stage's result counts as a win or loss for the episode.

This script evaluates the full mission chain, whereas checkpoint_test.py
evaluates a single model on one difficulty level.

Compared to curriculum_evaluation.py, this script is more manual:
  - Model paths and docking thresholds are set directly at the top of the file
    rather than automatically from a checkpoint folder.
  - No fixed random seed, so each run produces different starting conditions.
  - Does not suppress environment print statements (WIN!, etc.).

curriculum_evaluation.py is better suited for automated or repeatable
evaluation runs. This script is better suited for quickly testing a specific
set of checkpoints by hand.

Update the model paths at the top of the script to match your run folder
before running.
"""

import copy
import numpy as np
from pathlib import Path
from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from stable_baselines3 import PPO


def get_safe_action(env, action, obs):
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


# Build the checkpoint path relative to this file so the script works
# regardless of which directory it is launched from.
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"

# Load each curriculum stage model. Update these paths to match your run folder.
# Each model was trained on a progressively harder range of starting distances.
# Format: safe_ppo_model_{stage}_{run}_{epoch}.zip
# Example: safe_ppo_model_6_1_0.zip -> stage 6, run 1, epoch 0
model_1 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_5" /
                       "safe_ppo_model_1_5_0.zip"))
model_2 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_5" /
                       "safe_ppo_model_3_5_3.zip"))
model_3 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_5" /
                       "safe_ppo_model_6_5_24.zip"))
model_4 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_5" /
                       "safe_ppo_model_7_5_6.zip"))
model_5 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_5" /
                       "safe_ppo_model_8_5_1.zip"))

fuels = []  # total fuel used per episode
wins = []   # 1 for a successful dock, 0 for a failure

for i in range(100):
    print(i)
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

    while not done:
        # obs[:-1] *= np.random.uniform(0.99, 1.01, 1)
        if env.env.state[6] >= 10:
            env.env.state[6] = 0

        if env.is_drifting:
            # During a drift period the agent holds position with zero thrust.
            action = np.array([0.0, 0.0, 0.0])
        else:
            # Slice obs to match the observation size the model was trained on.
            # Older models expect 6 elements; newer ones expect 7 (with timestep).
            expected_obs_size = model.observation_space.shape[0]
            exp_action = model.predict(obs[:expected_obs_size],
                                    deterministic=True)[0]
            action = get_safe_action(env, exp_action, obs)  # TODO Check
        action *= np.random.uniform(0.95, 1.05)

        # Accumulate fuel as the sum of absolute thrust across all axes,
        # divided by mass and multiplied by step length to get delta-V units.
        epi_fuel += float(np.sum(np.abs(action))) / env.env.m * env.env.step_len

        obs, reward, term, trunc, info = env.step(action)
        # print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        done = term or trunc

        if done:
            if env.env.is_docked():
                print('docked')
            else:
                print('model: ', model_num, ' fail')

            if model_num == 1:
                # The easiest model just finished, so this is the final result
                # for the whole episode. Record win or loss and total fuel used.
                wins.append(1) if env.env.is_docked() else wins.append(0)
                fuels.append(epi_fuel)
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
                    print('model 4')
                    model = model_4
                    env.env.dock_dist = 50
                    env.env.dock_speed = 0.30
                elif model_num == 3:
                    print('model 3')
                    model = model_3
                    env.env.dock_dist = 10
                    env.env.dock_speed = 0.22
                elif model_num == 2:
                    print('model 2')
                    model = model_2
                    env.env.dock_dist = 2.5
                    env.env.dock_speed = 0.2
                else:
                    print('model 1')
                    model = model_1
                    env.env.dock_dist = 0.5
                    env.env.dock_speed = 0.2

# Print a summary of results across all episodes.
print(f'\nResults over {len(wins)} episodes:')
print(f'  Win rate:  {sum(wins)}/{len(wins)} ({100 * np.mean(wins):.1f}%)')

print(f'\nFuel (delta-V):')
print(f'  Mean:      {np.mean(fuels):.3f}')
print(f'  Median:    {np.median(fuels):.3f}')
print(f'  Min:       {min(fuels):.3f}')
print(f'  Max:       {max(fuels):.3f}')
print(f'  25th pct:  {np.percentile(fuels, 25):.3f}')
print(f'  50th pct:  {np.percentile(fuels, 50):.3f}')
print(f'  75th pct:  {np.percentile(fuels, 75):.3f}')