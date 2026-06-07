from drift_env import DriftEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def rl_train():
    env = DriftEnv(
        min_init_pos_bound=0.5,
        max_init_pos_bound=1
    )
    eval_callback = CheckpointCallback(save_freq=100_000,
                                       save_path='data/checkpoints',
                                       name_prefix='ppo_1m')
    # model = PPO.load(
    #     "data/checkpoints/ppo_1m_100000_steps.zip",
    #     env=env,
    #     verbose=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(1_000_000, callback=eval_callback)


if __name__ == '__main__':
    rl_train()
