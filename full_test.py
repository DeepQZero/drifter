import numpy as np

from drift_env import DriftEnv2
from initial_trainer import get_curriculum
from stable_baselines3 import PPO

import copy


model_1 = PPO.load("data/checkpoints/safe_ppo_model_4_1.zip")
model_2 = PPO.load("data/checkpoints/safe_ppo_model_7_1.zip")
model_3 = PPO.load("data/checkpoints/safe_ppo_model_9_1.zip")


fuels = []
wins = []

for _ in range(100):
    epi_fuel = 0
    curriculum, _ = get_curriculum(10)
    env = DriftEnv2(**curriculum)
    obs, info = env.reset()
    done = False

    model_num = 3
    model = model_3

    while not done:
        action = np.array([0.0, 0.0, 0.0]) if env.is_drifting else model.predict(obs)[0]
        epi_fuel += np.linalg.norm(action)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        if done:
            if model_num == 1:
                wins.append(1) if env.env.is_docked() else wins.append(0)
                fuels.append(epi_fuel)
            else:
                done = False
                model_num -= 1
                model = model_2 if model_num == 2 else model_1
                env.env.docking_pos_thresh = 10 if model_num == 2 else 0.5
                env.env.docking_neg_thresh = 0.2 if model_num == 2 else 0.2
                env.is_drifting = False

print(wins, np.mean(wins))
print(sorted(fuels, reverse=True), np.median(fuels), np.mean(fuels))

