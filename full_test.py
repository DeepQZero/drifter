import numpy as np

from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from stable_baselines3 import PPO

import copy


model_1 = PPO.load("data/checkpoints/safe_ppo_model_1_7.zip")
model_2 = PPO.load("data/checkpoints/safe_ppo_model_3_7.zip")
model_3 = PPO.load("data/checkpoints/safe_ppo_model_6_7.zip")
model_4 = PPO.load("data/checkpoints/safe_ppo_model_9_7.zip")


fuels = []
wins = []

for i in range(100):
    print(i)
    print('starting at model 4')
    epi_fuel = 0
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)
    obs, info = env.reset()
    done = False

    model_num = 4
    model = model_4
    t_step = 0
    while not done:
        if env.is_drifting or (t_step % 4) in list(range(5, 10)): # TODO
            action = np.array([0.0, 0.0, 0.0])
        else:
            action = model.predict(obs)[0]
        epi_fuel += np.linalg.norm(action)
        obs, reward, term, trunc, info = env.step(action)
        t_step += 1
        print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        done = term or trunc
        if done:
            if model_num == 1:
                wins.append(1) if env.env.is_docked() else wins.append(0)
                fuels.append(epi_fuel)
            else:
                done = False
                env.is_drifting = False
                t_step = 0
                model_num -= 1
                if model_num == 3:
                    model = model_3
                    env.env.docking_pos_thresh = 10
                    env.env.docking_speed_thresh = 0.22
                elif model_num == 2:
                    model = model_2
                    env.env.docking_pos_thresh = 2.5
                    env.env.docking_speed_thresh = 0.2
                else:
                    model = model_1
                    env.env.docking_pos_thresh = 0.5
                    env.env.docking_speed_thresh = 0.2


print(wins, np.mean(wins))
print(sorted(fuels, reverse=True), np.median(fuels), np.mean(fuels))
print(sorted(fuels, reverse=False)[24],
      sorted(fuels, reverse=False)[50],
      sorted(fuels, reverse=False)[75])