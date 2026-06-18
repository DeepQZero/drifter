import numpy as np
from pathlib import Path

from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from stable_baselines3 import PPO

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "data" / "checkpoints"

model_1 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_3" / "safe_ppo_model_1_3.zip"))
model_2 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_3" / "safe_ppo_model_3_3.zip"))
model_3 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_3" / "safe_ppo_model_6_3.zip"))
model_4 = PPO.load(str(CHECKPOINT_DIR / "safe_PPO_3" / "safe_ppo_model_8_3.zip"))

fuels = []
wins = []

for i in range(100):
    print(i)
    print('starting at model 4')
    epi_fuel = 0
    # Create a fresh curriculum and environment for each run.
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)
    obs, info = env.reset()
    done = False

    model_num = 4
    model = model_4
    t_step = 0
    while not done:
        # Hold position during drift or during the scheduled no-action window.
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
            # Only score the episode after the final model completes.
            if model_num == 1:
                wins.append(1) if env.env.is_docked() else wins.append(0)
                fuels.append(epi_fuel)
            else:
                # Step down through the model curriculum and continue the same run.
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
print(sorted(fuels, reverse=False)[25],
      sorted(fuels, reverse=False)[50],
      sorted(fuels, reverse=False)[75])
    