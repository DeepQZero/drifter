from drift_env import DriftEnv

from stable_baselines3 import PPO
import numpy as np

model = PPO.load(
    "data/checkpoints/ppo_1m_200000_steps.zip")

env = DriftEnv(
    min_init_pos_bound=0.5,
    max_init_pos_bound=1
)
win_count = 0
episode_time_steps = []
fuels = []

NUM_EPISODES = 100
for i in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    num_time_steps = 0
    episode_fuel = 0
    print('STARTING DISTANCE: ', np.linalg.norm(obs[0:3]))
    while not done:
        num_time_steps += 1
        action, _ = model.predict(obs, deterministic=True)
        episode_fuel += np.linalg.norm(action)
        obs, reward, term, trunc, info = env.step(action)
        print(obs)
        done = term or trunc
        if done:
            if reward > 0.99:
                win_count += 1
                print('Win')
            else:
                print('loss')
            episode_time_steps.append(num_time_steps)
            fuels.append(episode_fuel)

print(win_count)
print(np.mean(episode_time_steps), np.std(episode_time_steps))
print(np.mean(fuels), np.std(fuels))
