from stable_baselines3 import PPO
import numpy as np

obs = np.array([-8.02111213e+00,  4.74303247e+01, -1.32391300e+01,
                -8.34240908e-03, -1.83792687e-01, -6.94425887e-02,
                0.00000000e+00])

model = PPO.load("data/checkpoints/safe_ppo_model_2_6_9.zip")

action = model.predict(obs, deterministic=True)[0]

print(action)