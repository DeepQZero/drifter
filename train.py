import os

from drift_env import DriftTrainEnv as DriftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def rl_train():
    """
    Train the PPO agent and save checkpoints.

    Returns:
        None: This function runs training, then saves the each model and
              the VecNormalize statistics.
    """
    # Create base environment
    env = DriftEnv()

    # Vectorized + normalized environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Get base directory
    base_dir = os.path.dirname(__file__)

    # Create a checkpoints directory if it does not already exist
    checkpoints_dir = os.path.join(base_dir, "data", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Find the next available run number to avoid overwriting previous runs
    run_num = 1
    while os.path.exists(os.path.join(checkpoints_dir, f"drifter_PPO_{run_num}")):
        run_num += 1

    # Create a directory for this training run
    model_dir = os.path.join(checkpoints_dir, f"drifter_PPO_{run_num}")
    os.makedirs(model_dir, exist_ok=True)

    # Save a checkpoint every 25,000 training steps
    eval_callback = CheckpointCallback(save_freq=25_000,
                                       save_path=model_dir,
                                       name_prefix='drifter_ppo')

    # Create the PPO agent and enable TensorBoard logging
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                n_steps=512,        # was 2048 (default); faster iteration on CPU
                batch_size=64,     
                tensorboard_log=os.path.join(model_dir, "tensorboard_logs"))

    # Train the PPO agent
    model.learn(total_timesteps=50_000, callback=eval_callback)

    # Save model + normalization stats
    model.save(os.path.join(model_dir, "final_model"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))


if __name__ == "__main__":
    rl_train()
