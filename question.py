from drift_env import DriftTestEnv
from initial_trainer import get_curriculum

import numpy as np

curriculum, _ = get_curriculum(10)
env = DriftTestEnv(**curriculum)

obs, info = env.reset()
action = -1 * obs[3:6]
print(obs, action, np.linalg.norm(obs[3:6]))
obs, _, _, _, _ = env.step(action)
print(obs, np.linalg.norm(obs[3:6]))