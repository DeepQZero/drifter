import numpy as np

from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from stable_baselines3 import PPO

import copy


# model_1 = PPO.load("data/checkpoints/safe_ppo_model_6_1_0.zip")
# model_2 = PPO.load("data/checkpoints/safe_ppo_model_6_3_4.zip")
# model_3 = PPO.load("data/checkpoints/safe_ppo_model_6_6_8.zip")
# model_4 = PPO.load("data/checkpoints/safe_ppo_model_6_7_1.zip")
# model_5 = PPO.load("data/checkpoints/safe_ppo_model_6_8_2.zip")

model_1 = PPO.load("../data/checkpoints/safe_ppo_model_9_1_0.zip")
model_2 = PPO.load("../data/checkpoints/safe_ppo_model_9_3_2.zip")
model_3 = PPO.load("../data/checkpoints/safe_ppo_model_9_6_7.zip")
model_4 = PPO.load("../data/checkpoints/safe_ppo_model_9_7_7.zip")
model_5 = PPO.load("../data/checkpoints/safe_ppo_model_9_8_8.zip")

fuels = []
wins = []

for i in range(100):
    print(i)
    print('starting at model 5')
    epi_fuel = 0
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)

    obs, info = env.reset()
    done = False
    model_num = 5
    model = model_5
    # env.env.state[3] = 0.0
    # env.env.state[4] = 0.0
    # env.env.state[5] = 0.0
    while not done:
        if env.is_drifting:
            action = np.array([0.0, 0.0, 0.0])
        else:
            action = model.predict(obs, deterministic=True)[0]
        epi_fuel += float(np.sum(np.abs(action)))/env.env.m * env.env.step_len
        obs, reward, term, trunc, info = env.step(action)
        # print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        done = term or trunc
        if done:
            if env.env.is_docked():
                print('docked')
            else:
                print('model: ', model_num, ' fail')
            if model_num == 1:
                wins.append(1) if env.env.is_docked() else wins.append(0)
                fuels.append(epi_fuel)
            else:
                done = False
                env.is_drifting = False
                env.env.state[6] = 0
                model_num -= 1
                # env.env.state[3] = 0.0
                # env.env.state[4] = 0.0
                # env.env.state[5] = 0.0
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


print(wins, np.mean(wins))
print(sorted(fuels, reverse=True), np.median(fuels), np.mean(fuels))
print(sorted(fuels, reverse=False)[24],
      sorted(fuels, reverse=False)[50],
      sorted(fuels, reverse=False)[75])