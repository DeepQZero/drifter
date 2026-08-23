"""
drift_initial_trainer.py

Curriculum training script for the drift-assisted docking agent (Drifter-Learn).

Trains a PPO agent through progressively harder stages. Each stage trains
in a loop until the dock rate hits its threshold, then advances using the
last saved checkpoint as a starting point.

- Stages go from very close, slow starts (stage 0) up to full range
  (stage 9). Stages 4 and up introduce drift, where the agent coasts to
  the dock instead of thrusting the whole way.
- test_model() runs after every save during training, and can also be
  called standalone from __main__ to evaluate one checkpoint.
- Checkpoint format: safe_ppo_model_{run}_{stage}_{epoch}.zip
  (example: safe_ppo_model_1_6_0.zip -> run 1, stage 6, epoch 0).

Set SEED to a fixed integer for reproducibility. This seeds both the
environment (Env.reset(seed=...), sets self.np_random) and PPO's own RNG
(network init, action sampling). Reproducibility depends on Gymnasium's
seeded behavior, which has changed across versions before; we use
gymnasium==1.2.3, so re-verify seeded runs after any upgrade.
"""

import drift_env
from drift_env import DriftTrainEnv
import typing as tt
import numpy as np
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# User config: change these values to adjust how training runs.

# RUN_NAME controls the checkpoint folder name under data/checkpoints.
RUN_NAME = "safe_PPO"

# CHECKPOINT_PREFIX controls the saved model filename prefix.
CHECKPOINT_PREFIX = "safe_ppo_model"

# Number of curriculum stages. Includes stage 10, a test-only config
# used for evaluation, not a trained stage. Unique to the drift agent.
NUM_STAGES = 10

# Set to 0 to train from scratch, or a higher number to resume from a
# specific stage. RESUME_FROM is ignored when START_STAGE is 0.
START_STAGE = 0
RESUME_FROM = None  # example: rf"data\checkpoints\{RUN_NAME}_6\{CHECKPOINT_PREFIX}_6_7_1.zip"

# Optional seed for reproducibility.
# True (an int): fixes randomness across training and evaluation.
# False (None): uses a random seed each run.
SEED = None

# Number of parallel environments for training. Higher uses more CPU
# cores. Each environment step is cheap, so this mainly speeds up
# rollout collection between policy updates.
N_ENVS = 6

# Rollout size collected per training epoch, in timesteps.
TIMESTEPS_PER_EPOCH = 25_000

# Number of episodes test_model() runs per evaluation.
TEST_EPISODES = 1_000

# True: keep every checkpoint from every epoch, so training curves and
# graphs can be reconstructed later.
# False: keep only the final passing checkpoint per stage.
SAVE_ALL_EPOCHS = True

# Maximum number of training epochs per stage before moving on, even if
# the score threshold has not been reached. Prevents a single stage from
# running indefinitely if the model stalls.
MAX_EPOCHS_PER_STAGE = 10

# Set a specific run number to use for this training run, for example 13.
# Useful for running two versions side by side for a paper, or naming a
# run something memorable for an ablation.
# Set to None to auto-increment from the highest existing RUN_NAME folder.
MANUAL_RUN_ID = None  # example: 13

# True: refuse to run when MANUAL_RUN_ID already exists on disk.
# False: allow overwriting an existing MANUAL_RUN_ID folder.
# Only applies when MANUAL_RUN_ID is set.
PREVENT_OVERWRITE = True


def curriculum_learn(model_id: int, run_dir: str, seed: int = SEED):
    """Train a drift-assisted docking agent through a series of curriculum stages.

    Each stage trains in a loop until the dock rate hits the stage threshold
    or MAX_EPOCHS_PER_STAGE is reached, then moves to the next stage starting
    from the last saved checkpoint.

    Args:
        model_id: Number used in checkpoint filenames to identify this run.
        run_dir: Folder where checkpoints are saved.
        seed: Optional seed for reproducibility.
    """
    # If resuming mid-curriculum, use the provided checkpoint as the starting
    # point. After the first stage completes, last_save_path takes over and
    # RESUME_FROM is no longer used.
    last_save_path = RESUME_FROM if START_STAGE > 0 else None
    total_timesteps = 0
    run_start = time.time()

    for curr in range(START_STAGE, NUM_STAGES):
        print(f'Starting stage {curr}')
        stage_start = time.time()
        stage_timesteps = 0

        configs, threshold = get_curriculum(curr)
        env = make_vec_env(lambda: DriftTrainEnv(**configs), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

        try:
            if curr == 0:
                # Build a new model from scratch for the first stage.
                model = PPO('MlpPolicy', env,
                            learning_rate=0.0003,
                            ent_coef=0.01,
                            gamma=1.00,
                            n_steps=512,
                            batch_size=64,
                            verbose=1,
                            seed=seed)
            else:
                # Load the last checkpoint from the previous stage and
                # keep training in the new, harder environment.
                model = PPO.load(last_save_path, env=env)

            score = 0
            epoch = -1
            prev_save_path = None  # Tracks the previous epoch's checkpoint for cleanup

            while score < threshold and epoch < MAX_EPOCHS_PER_STAGE - 1:
                epoch += 1
                # TODO: what happens if model diverges?
                model.learn(total_timesteps=TIMESTEPS_PER_EPOCH)
                stage_timesteps += TIMESTEPS_PER_EPOCH
                total_timesteps += TIMESTEPS_PER_EPOCH

                save_path = os.path.join(
                    run_dir,
                    f"{CHECKPOINT_PREFIX}_{model_id}_{curr}_{epoch}"
                )
                model.save(save_path)
                last_save_path = save_path
                score = test_model(save_path, curr, seed=seed)

                save_name = os.path.basename(save_path)
                print(f"  Epoch {epoch}: dock rate {score:.3f} ({100*score:.1f}%)  |  {save_name}")

                # If SAVE_ALL_EPOCHS is off, delete the previous epoch's checkpoint
                # now that we have a newer one. The final passing checkpoint is
                # always kept because deletion happens before the next save.
                if not SAVE_ALL_EPOCHS and prev_save_path is not None:
                    zip_path = prev_save_path + ".zip"
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

                prev_save_path = save_path

            if score < threshold:
                is_last_stage = (curr == NUM_STAGES - 1)
                status = "Training complete for final stage" if is_last_stage else "Moving on to next stage"
                print(f"  Stage {curr} hit epoch limit ({MAX_EPOCHS_PER_STAGE}) "
                    f"with dock rate {score:.3f} ({100*score:.1f}%). {status}.")

            stage_elapsed = time.time() - stage_start
            stage_min = int(stage_elapsed // 60)
            stage_sec = int(stage_elapsed % 60)
            stage_hrs = stage_elapsed / 3600
            print(f"Stage {curr} complete: {stage_timesteps:,} timesteps, "
                f"{stage_min} min {stage_sec} sec ({stage_hrs:.2f} hrs)")
        finally:
            # Always close subprocess environments, even if training or testing fails.
            env.close()

    total_elapsed = time.time() - run_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)
    total_hrs = total_elapsed / 3600
    stages_run = curr - START_STAGE + 1
    avg_elapsed = total_elapsed / max(1, stages_run)
    avg_min = int(avg_elapsed // 60)
    avg_sec = int(avg_elapsed % 60)
    print(f"\nTraining complete")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Total time:      {total_min} min {total_sec} sec ({total_hrs:.2f} hrs)")
    print(f"  Avg per stage:   {avg_min} min {avg_sec} sec")


def test_model(path: str, curriculum: int, seed: int = SEED) -> float:
    """Evaluate a checkpoint over 1000 episodes and return the dock rate.

    Called automatically during training after each save. Can also be run
    standalone from __main__ to evaluate a specific checkpoint.

    An episode counts as a success if the agent docks directly or reaches
    a state where drifting to the dock is possible (drift success).

    Args:
        path: Path to the saved model checkpoint.
        curriculum: Stage number, used to set up the right environment.
        seed: Optional seed for reproducibility.

    Returns:
        Fraction of episodes that ended in a successful dock or drift.
    """
    # test_model runs one episode at a time with a specific seed offset per
    # episode, so it stays a single environment.
    configs, _ = get_curriculum(curriculum)
    env = DriftTrainEnv(**configs)
    model = PPO.load(path, env=env)
    all_rews, all_fuels, all_docks = [], [], []

    for i in range(TEST_EPISODES):
        done = False
        # Offset by episode index so episodes stay varied while the whole
        # test set is reproducible when seed is fixed.
        episode_seed = None if seed is None else seed + i
        obs, info = env.reset(seed=episode_seed)
        epi_fuel, epi_reward = 0, 0
        is_drift = False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, term, trunc, info = env.step(action)
            epi_reward += reward
            # Fuel (delta-V): force magnitude / mass * step length, in m/s.
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

    print(f"  Model stats for stage {curriculum}:")
    print(f"    Dock rate:    {np.mean(all_docks):.3f} ({100*np.mean(all_docks):.1f}%)")
    print(f"    Mean reward:  {np.mean(all_rews):.5f}")
    print(f"    Fuel (delta-V):  mean {np.mean(all_fuels):.3f} m/s  "
          f"median {np.median(all_fuels):.3f} m/s")
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
        threshold = 0.98
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
        threshold = 0.98
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
        configs['max_episode_len'] = 6
        configs['max_lookahead_len'] = 10
        configs['max_boundary_box'] = 50
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 20
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 5: longer lookahead, wider start range.
    if curriculum >= 5:
        threshold = 0.95
        configs['max_episode_len'] = 6
        configs['max_lookahead_len'] = 25
        configs['max_boundary_box'] = 40
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 35
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 6: even wider start range and boundary.
    if curriculum >= 6:
        threshold = 0.98
        configs['max_episode_len'] = 6
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 60
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 50
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10

    # Stage 7: long range start, wide docking target, higher speed allowed.
    if curriculum >= 7:
        threshold = 0.98
        configs['pos_thresh'] = 50
        configs['speed_thresh'] = 0.3
        configs['max_episode_len'] = 8
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 110
        configs['min_init_pos_bound'] = 50
        configs['max_init_pos_bound'] = 100
        configs['max_init_vel_bound'] = 0.4  # TODO
        configs['drift_step_len'] = 10

    # Stage 8: full range with very wide docking target.
    if curriculum >= 8:
        threshold = 0.98
        configs['pos_thresh'] = 100
        configs['speed_thresh'] = 0.4
        configs['max_episode_len'] = 10
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

    if MANUAL_RUN_ID is not None:
        # Use the run number set in MANUAL_RUN_ID instead of auto-incrementing.
        # Useful for running paired experiments with predictable IDs, e.g.
        # safe_PPO_13 as a baseline and safe_PPO_14 as an ablation.
        run_num = MANUAL_RUN_ID
        run_dir = os.path.join(checkpoints_dir, f"{RUN_NAME}_{run_num}")

        if PREVENT_OVERWRITE and os.path.exists(run_dir):
            raise FileExistsError(
                f"{RUN_NAME}_{run_num} already exists at {run_dir}. "
                f"Set PREVENT_OVERWRITE = False to overwrite it, or choose "
                f"a different MANUAL_RUN_ID."
            )
    else:
        # Find the next available run number so old checkpoints are not overwritten.
        run_num = 1
        while os.path.exists(
            os.path.join(checkpoints_dir, f"{RUN_NAME}_{run_num}")
        ):
            run_num += 1
        run_dir = os.path.join(checkpoints_dir, f"{RUN_NAME}_{run_num}")

    os.makedirs(run_dir, exist_ok=True)
    model_id = run_num

    print(f"\nSaving models to: {run_dir}")
    print(f"Model ID: {model_id}")
    if SEED is not None:
        print(f"Seed: {SEED}")
    else:
        print("Seed: not set")

    # Record which SafeRL comparison options (drift_env.py) were active. All
    # default to off = current drifter behavior. Note the drift env already
    # enforces the velocity constraint unconditionally via is_unsafe, so it
    # has no separate velocity-constraint flag.
    print("SafeRL options:")
    print(f"  obs (speed + max_vel_limit):   {drift_env.SAFERL_OBS}")
    print(f"  exponential distance reward:   {drift_env.SAFERL_EXP_DIST_REWARD}")
    print(f"  delta-V fuel penalty:          {drift_env.SAFERL_DELTA_V_PENALTY}\n")
    curriculum_learn(model_id, run_dir)

    # To test a specific checkpoint after training:
    # test_model('data/checkpoints/safe_PPO_6/safe_ppo_model_6_7_1.zip', 7)
