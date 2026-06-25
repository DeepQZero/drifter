"""
initial_trainer.py

Curriculum training script for the drift-assisted docking agent (Drifter-Learn).

Trains a PPO agent through a series of progressively harder stages. Each stage
trains in a loop until the dock rate hits the stage threshold, then advances to
the next stage using the last saved checkpoint as a starting point.

Curriculum stages go from very close, slow starts (stage 0) up to full
range (stage 9). Stages 4 and above introduce drift mechanics, where the agent
learns to coast to the dock instead of thrusting the whole way.

test_model() is also called during training to measure dock rate after each
save. It can also be called standalone from __main__ to evaluate a specific
checkpoint without running a full training session.

Checkpoint naming format: safe_ppo_model_{stage}_{run}_{epoch}.zip
Example: safe_ppo_model_6_1_0.zip -> stage 6, run 1, epoch 0
"""

from drift_env import DriftTrainEnv
import typing as tt
import numpy as np
from stable_baselines3 import PPO
import os
import time

# User config: change these values to adjust how training runs.
# Set START_STAGE = 0 and RESUME_FROM = None to train from scratch.
# Set START_STAGE to a higher number and RESUME_FROM to a checkpoint path
# to resume training from a specific stage without redoing earlier ones.
START_STAGE = 0
RESUME_FROM = None  # example: r"data\checkpoints\safe_PPO_6\safe_ppo_model_7_6_1.zip"

# If True, every checkpoint from every epoch is kept on disk.
# If False, only the final passing checkpoint per stage is kept.
SAVE_ALL_EPOCHS = False


def curriculum_learn(model_id: int, run_dir: str):
    """Train a drift-assisted docking agent through a series of curriculum stages.

    Each stage trains in a loop until the dock rate hits the stage threshold,
    then moves to the next stage starting from the last saved checkpoint.

    Args:
        model_id: Number used in checkpoint filenames to identify this run.
        run_dir: Folder where checkpoints are saved.
    """
    # If resuming mid-curriculum, use the provided checkpoint as the starting
    # point. After the first stage completes, last_save_path takes over and
    # RESUME_FROM is no longer used.
    last_save_path = RESUME_FROM if START_STAGE > 0 else None
    total_timesteps = 0
    run_start = time.time()

    for curr in range(START_STAGE, 10):
        print('Starting Curriculum: ', curr)
        stage_start = time.time()
        stage_timesteps = 0

        configs, threshold = get_curriculum(curr)
        env = DriftTrainEnv(**configs)

        if curr == 0:
            # Build a new model from scratch for the first stage.
            model = PPO('MlpPolicy', env,
                        learning_rate=0.0003,
                        ent_coef=0.01,
                        gamma=1.00,
                        n_steps=512,
                        batch_size=64,
                        verbose=1)
        else:
            # Load the last checkpoint from the previous stage and
            # keep training in the new, harder environment.
            model = PPO.load(last_save_path, env=env)

        score = 0
        epoch = -1
        prev_save_path = None  # tracks the previous epoch's checkpoint for cleanup
        while score < threshold:
            epoch += 1
            # TODO: what happens if model diverges?
            model.learn(total_timesteps=25_000)
            stage_timesteps += 25_000
            total_timesteps += 25_000

            save_path = os.path.join(
                run_dir,
                f"safe_ppo_model_{curr}_{model_id}_{epoch}"
            )
            model.save(save_path)
            last_save_path = save_path
            score = test_model(save_path, curr)
            print('Saved Model: ', save_path, ' Score: ', score)

            # If SAVE_ALL_EPOCHS is off, delete the previous epoch's checkpoint
            # now that we have a newer one. The final passing checkpoint is
            # always kept because deletion happens before the next save.
            if not SAVE_ALL_EPOCHS and prev_save_path is not None:
                zip_path = prev_save_path + ".zip"
                if os.path.exists(zip_path):
                    os.remove(zip_path)

            prev_save_path = save_path

        stage_time = time.time() - stage_start
        print(f'Stage {curr} done: {stage_timesteps:,} timesteps, '
              f'{stage_time / 60:.1f} min')

    total_time = time.time() - run_start
    print(f'Training complete: {total_timesteps:,} total timesteps, '
          f'{total_time / 60:.1f} min total')


def test_model(path: str, curriculum: int) -> float:
    """Evaluate a checkpoint over 1000 episodes and return the dock rate.

    Called automatically during training after each save. Can also be run
    standalone from __main__ to evaluate a specific checkpoint.

    An episode counts as a success if the agent docks directly or reaches
    a state where drifting to the dock is possible (drift success).

    Args:
        path: Path to the saved model checkpoint.
        curriculum: Stage number, used to set up the right environment.

    Returns:
        Fraction of episodes that ended in a successful dock or drift.
    """
    configs, _ = get_curriculum(curriculum)
    env = DriftTrainEnv(**configs)
    model = PPO.load(path, env=env)
    all_rews, all_fuels, all_docks = [], [], []

    for i in range(1_000):
        done = False
        obs, info = env.reset()
        epi_fuel, epi_reward = 0, 0
        is_drift = False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, term, trunc, info = env.step(action)
            epi_reward += reward
            epi_fuel += np.linalg.norm(action) / env.env.m * env.env.step_len
            is_drift, _ = env.det_drift()
            done = term or trunc or is_drift
            if done:
                # Count as a success if the agent docked or is in a drift
                # position where coasting would complete the dock.
                if env.env.is_docked() or is_drift:
                    all_docks.append(1)
                else:
                    all_docks.append(0)
                all_rews.append(epi_reward)
                all_fuels.append(epi_fuel)

    print('Model Stats for Curriculum:', curriculum)
    print('Dock: ', np.mean(all_docks),
          'Reward: ', np.mean(all_rews),
          'Fuel: ', np.mean(all_fuels), np.median(all_fuels)
          )
    return float(np.mean(all_docks))


def get_curriculum(curriculum: int) -> tt.Tuple[dict, float]:
    """Return environment settings and a pass threshold for a given stage.

    Stages progress from easy (close start, small boundary) to hard (far
    start, full range). Stages 4 and above set drift_step_len > 1,
    which activates the drift mechanic in the environment.

    Stage 10 is the test environment config used during evaluation, not
    a training stage.

    Args:
        curriculum: Stage number, 0 through 10.

    Returns:
        A tuple of (environment config dict, pass threshold float).
    """
    # Default config. Individual stages override specific values below.
    configs = {
        'fixed_start': False,
        'fixed_state': np.array([100, 0, 0, 0, 0, 0, 0]),  # TODO change
        'reward_structure': "dense",
        'max_episode_len': 10_000,
        'max_lookahead_len': 1_000,
        'max_boundary_box': 200.0,
        'max_control': 1.0,
        'max_total_dv': 1_000.0,
        'pos_thresh': 0.5,
        'speed_thresh': 0.2,
        'min_init_pos_bound': 100.0,
        'max_init_pos_bound': 150.0,
        'max_init_vel_bound': 0.5,
        'step_len': 1,
        'fuel_used': None,
        'time_step': None,
        'drift_step_len': 1
    }
    threshold = 0.99

    # Stage 0: very close start. Agent learns basic thrust toward target.
    if curriculum >= 0:
        threshold = 0.95
        configs['pos_thresh'] = 0.5
        configs['speed_thresh'] = 0.2
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 20
        configs['max_boundary_box'] = 3
        configs['min_init_pos_bound'] = 0.5
        configs['max_init_pos_bound'] = 1
        configs['max_init_vel_bound'] = 0.2

    # Stage 1: slightly farther start and larger boundary.
    if curriculum >= 1:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 10
        configs['min_init_pos_bound'] = 0.5
        configs['max_init_pos_bound'] = 2.5
        configs['max_init_vel_bound'] = 0.2

    # Stage 2: wider docking target to help the agent succeed from farther away.
    if curriculum >= 2:
        threshold = 0.95
        configs['pos_thresh'] = 2.5
        configs['speed_thresh'] = 0.2
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 20
        configs['min_init_pos_bound'] = 2.5
        configs['max_init_pos_bound'] = 5
        configs['max_init_vel_bound'] = 0.2

    # Stage 3: longer lookahead and bigger space.
    if curriculum >= 3:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 100
        configs['max_boundary_box'] = 30
        configs['min_init_pos_bound'] = 2.5
        configs['max_init_pos_bound'] = 10
        configs['max_init_vel_bound'] = 0.2

    # Stage 4: drift mechanic introduced (drift_step_len = 10).
    # Wide docking target at medium range, slightly higher speed allowed.
    if curriculum >= 4:
        threshold = 0.95
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.22
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 10
        configs['max_boundary_box'] = 50
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 20
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 5: longer lookahead, wider start range.
    if curriculum >= 5:
        threshold = 0.95
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 25
        configs['max_boundary_box'] = 40
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 35
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 6: even wider start range and boundary.
    if curriculum >= 6:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 60
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 50
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 7: long range start, wide docking target, higher speed allowed.
    if curriculum >= 7:
        threshold = 0.985
        configs['pos_thresh'] = 50
        configs['speed_thresh'] = 0.3
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 110
        configs['min_init_pos_bound'] = 50
        configs['max_init_pos_bound'] = 100
        configs['max_init_vel_bound'] = 0.4  # TODO
        configs['drift_step_len'] = 10

    # Stage 8: full range with very wide docking target.
    if curriculum >= 8:
        threshold = 0.985
        configs['pos_thresh'] = 100
        configs['speed_thresh'] = 0.4
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 200
        configs['min_init_pos_bound'] = 100
        configs['max_init_pos_bound'] = 150
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 10

    # Stage 9: strict docking target at full mission range.
    # if curriculum >= 9:
    #     threshold = 0.985
    #     configs['pos_thresh'] = 50
    #     configs['speed_thresh'] = 0.3
    #     configs['max_episode_len'] = 5
    #     configs['max_lookahead_len'] = 100
    #     configs['max_boundary_box'] = 200
    #     configs['min_init_pos_bound'] = 100
    #     configs['max_init_pos_bound'] = 150
    #     configs['max_init_vel_bound'] = 0.5
    #     configs['drift_step_len'] = 10

    if curriculum >= 10:  # Testing curriculum, not a training stage.
        configs['pos_thresh'] = 100
        configs['speed_thresh'] = 0.4
        configs['max_episode_len'] = 9_000  # TODO think about changing
        configs['max_lookahead_len'] = 1000
        configs['max_boundary_box'] = 200
        configs['min_init_pos_bound'] = 100
        configs['max_init_pos_bound'] = 150
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 1

    return configs, threshold


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    checkpoints_dir = os.path.join(
        base_dir,
        "data",
        "checkpoints"
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Find the next available run number so old checkpoints are not overwritten.
    run_num = 1
    while os.path.exists(
        os.path.join(checkpoints_dir, f"safe_PPO_{run_num}")
    ):
        run_num += 1

    run_dir = os.path.join(
        checkpoints_dir,
        f"safe_PPO_{run_num}"
    )
    os.makedirs(run_dir, exist_ok=True)
    model_id = run_num

    print(f"\nSaving models to: {run_dir}")
    print(f"Model ID: {model_id}\n")
    curriculum_learn(model_id, run_dir)

    # To test a specific checkpoint after training:
    # test_model('data/checkpoints/safe_PPO_6/safe_ppo_model_6_7_1.zip', 7)
