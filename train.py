from docking_env import SpaceCraftDockingEnv3D as TheEnvironment

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def rl_train():
    env = TheEnvironment()
    eval_callback = CheckpointCallback(save_freq=100_000,
                                       save_path='data/ppo_logs/forward/models_docking/checkpoints',
                                       name_prefix='ppo_docking2')
    model = PPO.load(
        "data/ppo_logs/forward/models_docking/checkpoints"
        "/ppo_docking_1000000_steps.zip",
        env=env,
        verbose=1)
    # model = PPO("MlpPolicy", env, verbose=1)
    #             # tensorboard_log="../data/ppo_logs/forward/models_docking/")
    model.learn(1_000_000, callback=eval_callback)


if __name__ == '__main__':
    rl_train()
