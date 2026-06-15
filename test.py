# TODO docking but not drifting is not worth anything. This doesn't work.
from drift_env import DriftTrainEnv
from initial_trainer import get_curriculum

from stable_baselines3 import PPO
import numpy as np

model = PPO.load(
    "data/checkpoints/safe_ppo_model_7_6_4.zip")

curriculum, _ = get_curriculum(6)
env = DriftTrainEnv(**curriculum)

win_count = 0
episode_time_steps = []
fuels = []

NUM_EPISODES = 1000
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
        done = term or trunc
        print(done)
        if done:
            print(reward)
            if reward > 9:
                win_count += 1
                print('Win')
            else:
                print('loss')
            episode_time_steps.append(num_time_steps)
            fuels.append(episode_fuel)

print(win_count)
print(np.mean(episode_time_steps), np.std(episode_time_steps))
print(np.mean(fuels), np.std(fuels))