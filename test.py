from docking_env import SpaceCraftDockingEnv3D

from stable_baselines3 import PPO

import numpy as np

model = PPO.load(
    "../data/ppo_logs/forward/models_docking/checkpoints"
    "/ppo_model_trial1_1000000_steps.zip")

env = SpaceCraftDockingEnv3D(reward_structure='sparse')
win_count = 0
episode_time_steps = []
fuels = []

NUM_EPISODES = 100
for i in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    num_time_steps = 0
    while not done:
        num_time_steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        if done:
            if reward > 0.99:
                win_count += 1
            episode_time_steps.append(num_time_steps)
print(win_count)
print(np.mean(episode_time_steps), np.std(episode_time_steps))
