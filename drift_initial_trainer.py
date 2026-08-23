"""Curriculum trainer for the drift-assisted docking agent.

How training works:
- The trainer moves through stages, from easy tasks to harder tasks.
- Each stage keeps training until it reaches its dock-rate target.
- After a stage passes, training continues from the latest saved checkpoint.

Stage behavior:
- Every stage can end with a successful coast after a thrust action.
- Stages 0 through 3 use 1-second coast steps; stages 4 through 8 use
    10-second coast steps to extend the lookahead.
- Stage 9 is evaluation-only and returns to 1-second coast steps.

Important:
- No single stage is meant to be used on its own.
- The full curriculum chain is what the drift evaluation scripts use.
- The function resolve_drift_model_chain() resolves that full chain.

Evaluation:
- test_model() runs after each save.
- It can also be called on its own from __main__ to score any checkpoint.

Checkpoint format:
- Files use this naming pattern: safe_ppo_model_{run}_{stage}_{epoch}.zip
- The final stage is also copied to final_stage_model_safe_{run}.zip

Reproducibility:
- Set SEED to make a run repeatable.
- This seed controls the environment, PPO network setup, and action sampling.
"""

import drift_env
from drift_env import DriftTrainEnv, DriftDockingEnv3D
import typing as tt
import numpy as np
import torch
from stable_baselines3 import PPO
import os
import csv
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from evaluation_utilities import (
    save_run_metadata, group_env_flags, get_package_versions, wrap_with_action_noise,
    format_duration, clarify_ppo_kl_early_stop_message, apply_flag_config_overrides,
    cap_blas_threads_per_worker,
)

# Rewrites PPO's early-stop KL warning to show the actual trigger (1.5x target_kl).
clarify_ppo_kl_early_stop_message()

# Faster closed-form CWH propagation: same results as solve_ivp, ~5-11x faster.
drift_env.FAST_ANALYTIC_PROPAGATION = True

# USER CONFIG:
# Edit anything below to change how a run behaves. Grouped so the
# most frequently changed settings come first.

# --- Run identity: what this run is called on disk ---

# True: refuse to run if MANUAL_RUN_ID already exists on disk, to avoid
# accidentally overwriting an old run's checkpoints.
# False: allow overwriting it. Only applies when MANUAL_RUN_ID is set.
PREVENT_OVERWRITE = True

# Checkpoint folder name under data/checkpoints, e.g. data/checkpoints/safe_PPO_7.
RUN_NAME = "safe_PPO"

# Prefix for saved model filenames, e.g. safe_ppo_model_7_3_2.zip.
CHECKPOINT_PREFIX = "safe_ppo_model"

# Pick a specific run number instead of auto-incrementing. Useful for
# pairing two runs, e.g. a baseline and an ablation, under predictable
# and matching numbers.
# None: use the next number after the highest existing RUN_NAME folder.
# An int, e.g. 13: use exactly that run number.
MANUAL_RUN_ID = None

# --- What to train ---

# Stages 0 through 8, easiest to hardest. Stage 0 starts close and slow;
# stage 8 is full range. Stages 4 through 8 use 10-second coast steps.
# Stage 9 is an evaluation-only config with 1-second coast steps.
NUM_STAGES = 9

# Which stage to start training at. Use 0 to train from scratch.
# RESUME_FROM only applies when START_STAGE is above 0.
START_STAGE = 0

# The checkpoint to resume from, e.g.
# RESUME_FROM = rf"data\checkpoints\{RUN_NAME}_6\{CHECKPOINT_PREFIX}_6_7_1.zip"
RESUME_FROM = None

# Seed for training randomness: network initialization, action sampling,
# and start states.
# An int, e.g. 100: reproducible run.
# None: pick a random seed and print it, so the run can still be reproduced later.
SEED = None

# Seed for the pass/fail check after each epoch. Fixed and kept separate
# from SEED, so every run is scored on the same set of episodes.
EVAL_SEED = 0

# --- Training budget and compute ---

# Number of parallel environments. Each runs in its own process, so this
# is roughly how many CPU cores training uses. Mainly speeds up
# collecting rollouts between policy updates.
N_ENVS = 5

# Limit on PyTorch's thread pool, so it doesn't compete with the
# N_ENVS worker processes for CPU cores.
N_TORCH_THREADS = 2

# Timesteps collected per training epoch before the policy is evaluated
# and saved again.
TIMESTEPS_PER_EPOCH = 25_000

# Give up on a stage after this many epochs, even if it never reaches
# its pass threshold, so a stalled stage doesn't run forever.
MAX_EPOCHS_PER_STAGE = 10

# Episodes test_model() runs to score each epoch's checkpoint.
TEST_EPISODES = 1_000

# --- Environment flags: the main ablation options ---
# Flip one to run an ablation. Each is saved to run_metadata.json, so
# the eval scripts label a single-flag change automatically. Drift's
# clean baseline is all-off: it reaches a high dock rate without any
# extra reward shaping.

# Master toggles. Override every flag in one family at once.
# True: turn on every SAFERL_* flag below.
# False: pick flags individually below, as normal.
SAFERL_ALL = False

# Reproduces SafeRL (2022)'s docking setup: every SAFERL_* flag, their
# 8-element observation, their tuned gamma/GAE lambda, and normalized
# observations.
# True: match SafeRL's setup as closely as this project's flags allow.
# False: SAFERL_ALL and the individual flags below behave as normal.
SAFERL_EXACT = False

# True: turn on every SPACE_CONTROLS_* flag below.
# False: pick flags individually below, as normal.
SPACE_CONTROLS_ALL = False

# Reproduces DRL for Space Controls (2024)'s docking setup: every
# SPACE_CONTROLS_* flag plus the SafeRL-derived reward terms their paper
# reuses (distance reward, Delta-v penalty).
# True: match their setup as closely as this project's flags allow.
# False: SPACE_CONTROLS_ALL and the individual flags below behave as normal.
SPACE_CONTROLS_EXACT = True

# Warn if both exact setups are requested, since they conflict and neither is a clean baseline.
if SAFERL_EXACT and SPACE_CONTROLS_EXACT:
    print(
        "WARNING: SAFERL_EXACT and SPACE_CONTROLS_EXACT are both True. Each "
        "replicates a different paper's setup; running both together trains "
        "a combined config that matches neither paper's baseline."
    )

# SafeRL (2022) flags:

# True: appends [speed, max_vel_limit] to the observation.
# False: keeps the plain 7-element observation.
SAFERL_OBS = False

# True: uses an exponential distance reward, which pulls hardest near
# the target.
# False: uses a linear approach reward instead.
SAFERL_EXP_DIST_REWARD = False

# True: charges a Delta-v fuel penalty on every step.
# False: thrust is free, so nothing discourages full throttle.
SAFERL_DELTA_V_PENALTY = False

# True: adds up to +1 extra reward for winning early, whether that's a
# direct dock or a coast win.
# False: uses a flat win reward with no time bonus.
SAFERL_SUCCESS_TIME_BONUS = False

# True: uses a budgeted velocity constraint. Violations build up a
# penalty, and only end the episode once the budget runs out.
# False: the very first violation ends the episode immediately.
SAFERL_VEL_CONSTRAINT = False

# True: uses SafeRL's flat +2 win reward instead of this project's own +1.
# False: keeps this project's own +1 win reward.
SAFERL_SUCCESS_REWARD = False

# True: goes out of bounds at SafeRL's 40,000 m, i.e. almost never.
# False: uses max_boundary_box, normally 200 m.
SAFERL_MAX_DISTANCE = False

# True: removes the time penalty entirely, matching SafeRL's reward.
# False: keeps this environment's default time penalty (unless
# PAPER_TIME_PENALTY or SPACE_CONTROLS_TIME_PENALTY overrides it instead).
SAFERL_NO_TIME_PENALTY = False

# DRL for Space Controls (2024) flags (arXiv 2405.12355):

# True: drops the timestep from the observation, leaving 6 elements.
# False: keeps all 7.
SPACE_CONTROLS_OBS = False

# True: uses a flat -0.01 time penalty per step, roughly 20x this
# project's own.
# False: uses this environment's default time penalty.
SPACE_CONTROLS_TIME_PENALTY = True

# True: limits the starting speed at 80% of the speed limit.
# False: allows starting speeds up to 100% of the limit.
SPACE_CONTROLS_INIT_VEL_FRACTION = True

# True: measures the speed limit from the docking radius (their Eq. 5).
# False: measures it from the origin instead (SafeRL 2022 Eq. 17).
SPACE_CONTROLS_SPEED_LIMIT_OFFSET = True

# True: uses their velocity penalty, which never ends the episode on its own.
# False: uses SAFERL_VEL_CONSTRAINT instead, or no constraint at all.
SPACE_CONTROLS_VEL_CONSTRAINT = True

# True: docks at their looser 10 m radius.
# False: uses pos_thresh, normally 0.5 m.
SPACE_CONTROLS_DOCK_RADIUS = False

# True: goes out of bounds at their 800 m.
# False: uses max_boundary_box, normally 200 m.
SPACE_CONTROLS_MAX_DISTANCE = True

# True: limits thrust at 0.1 N per axis, their low-thrust setting.
# False: uses max_control, normally 1.0 N.
SPACE_CONTROLS_LOW_THRUST = False

# This project's own flags, with no SafeRL equivalent:

# True: adds a braking margin (distance minus stopping distance) to the
# observation, giving the agent an explicit "start braking now" signal.
# False: leaves the observation unchanged.
CUSTOM_BRAKING_MARGIN_OBS = False

# True: penalizes going over the speed limit, scaled by how close the
# agent is to the target. Pair with SAFERL_VEL_CONSTRAINT, or the first
# violation ends the episode before the penalty has a chance to teach
# anything.
# False: no extra speed penalty.
CUSTOM_VEL_PENALTY = False

# True: scales observations to a similar range (positions / 100,
# velocities / 0.5, timestep / episode length).
# False: uses raw, unscaled values.
NORMALIZE_OBS = True

# Flags matching this project's paper's stated Methods text, added
# to test whether the paper's stated value or this environment's actual
# default is the one that's right.

# True: uses a flat -0.0005 per-step time penalty, the paper's stated value.
# False: uses this environment's own -0.005 default.
PAPER_TIME_PENALTY = False

# True: sets the unsafe-termination reward to 0, the paper's stated value.
# False: uses this environment's own -1 default (the same as a crash).
PAPER_UNSAFE_REWARD = False

# --- Training-side ablations ---
# These change how training runs rather than how the environment works,
# so run_metadata.json records them separately from ENV_FLAGS.

# True: sparse reward, +1 for docking and nothing else. Tests whether
#       the task is learnable with no shaping at all.
# False: the usual dense shaped reward.
SPARSE_REWARD = False

# Adds noise to actions during training, so the policy learns to be
# robust to imperfect thrusters. The eval scripts have their own
# USE_ACTION_NOISE for test time.
# True: train on noisy actions.
# False: train on exact, noise-free actions.
TRAIN_ACTION_NOISE = False

# True: uses SafeRL (2022)'s tuned discount and GAE lambda instead of
# this file's own gamma=1.00 (undiscounted returns).
# False: uses this file's PPO hyperparameters.
SAFERL_GAMMA_LAMBDA = False

# How much noise to add, as a fraction, e.g. 0.05 for +/-5%. Matches the
# eval scripts' noise magnitude, so training and testing agree.
TRAIN_ACTION_NOISE_FRACTION = 0.05

SAFERL_GAMMA_VALUE = 0.968559
SAFERL_LAMBDA_VALUE = 0.928544

# --- Hyperparameters ---

# PPO's learning rate.
LEARNING_RATE = 0.0003

# Entropy bonus.
ENT_COEF = 0.01

# Limits how far one PPO update can move the policy, measured as approx_kl.
# None uses PPO's default (no limit); drift has not needed one.
TARGET_KL = None

# Discounted instead of 1.00. SAFERL_GAMMA_LAMBDA swaps this and
# GAE_LAMBDA for SafeRL's tuned values. SAFERL_EXACT implies it too, so
# check run_metadata.json's "gamma"/"gae_lambda" for what ran,
# not just this flag.
GAMMA = SAFERL_GAMMA_VALUE if (SAFERL_GAMMA_LAMBDA or SAFERL_EXACT) else 1.00
GAE_LAMBDA = SAFERL_LAMBDA_VALUE if (SAFERL_GAMMA_LAMBDA or SAFERL_EXACT) else 0.95

# --- Stage-advance behavior ---

# Prevents exploration collapse, where early stages learn a very narrow
# policy with little randomness left, making later, wider stages harder to
# explore.
# True: resets the policy's exploration noise (log_std) at each stage
# advance, so the next stage starts with fresh exploration in the larger
# environment instead of reusing a narrow, already-converged policy.
# False: keeps the exploration noise learned in the previous stage.
RESET_STD_ON_STAGE_ADVANCE = True

# What log_std to reset to when the flag above is on. 0.0 matches PPO's
# default (std = 1.0), the same exploration level stage 0 starts with.
RESET_STD_LOG_VALUE = 0.0

# --- Saved output ---

# True: keeps every epoch's checkpoint, so training curves can be
# rebuilt later from disk.
# False: keeps only the final passing checkpoint for each stage.
SAVE_ALL_EPOCHS = True

# Controls how TensorBoard logs are organized. Doesn't change training.
# True: one fresh log per epoch, so epochs can be overlaid and compared
# in TensorBoard.
# False: one continuous log per stage, all epochs merged into one curve.
TENSORBOARD_RESET_NUM_TIMESTEPS = False

# One dict for all env/reward toggles: SafeRL, Space Controls, custom,
# and paper flags. Family ALL/EXACT switches are ORed in here; normalization
# and the custom brake override stay as special cases.
ENV_FLAGS = {
    "saferl_obs": SAFERL_OBS or SAFERL_ALL or SAFERL_EXACT or SPACE_CONTROLS_EXACT,
    "saferl_exp_dist_reward": SAFERL_EXP_DIST_REWARD or SAFERL_ALL or SAFERL_EXACT or SPACE_CONTROLS_EXACT,
    "saferl_delta_v_penalty": SAFERL_DELTA_V_PENALTY or SAFERL_ALL or SAFERL_EXACT or SPACE_CONTROLS_EXACT,
    "saferl_success_time_bonus": SAFERL_SUCCESS_TIME_BONUS or SAFERL_ALL or SAFERL_EXACT,
    "saferl_success_reward": SAFERL_SUCCESS_REWARD or SAFERL_ALL or SAFERL_EXACT,
    "saferl_vel_constraint": SAFERL_VEL_CONSTRAINT or SAFERL_ALL or SAFERL_EXACT,
    "saferl_max_distance": SAFERL_MAX_DISTANCE or SAFERL_ALL or SAFERL_EXACT,
    "saferl_no_time_penalty": SAFERL_NO_TIME_PENALTY or SAFERL_ALL or SAFERL_EXACT,
    "space_controls_obs": SPACE_CONTROLS_OBS or SPACE_CONTROLS_ALL or SAFERL_EXACT or SPACE_CONTROLS_EXACT,
    "space_controls_time_penalty": SPACE_CONTROLS_TIME_PENALTY or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_init_vel_fraction": SPACE_CONTROLS_INIT_VEL_FRACTION or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_speed_limit_offset": SPACE_CONTROLS_SPEED_LIMIT_OFFSET or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_vel_constraint": SPACE_CONTROLS_VEL_CONSTRAINT or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_dock_radius": SPACE_CONTROLS_DOCK_RADIUS or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_max_distance": SPACE_CONTROLS_MAX_DISTANCE or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "space_controls_low_thrust": SPACE_CONTROLS_LOW_THRUST or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "custom_braking_margin_obs": False if (SAFERL_EXACT or SPACE_CONTROLS_EXACT) else CUSTOM_BRAKING_MARGIN_OBS,
    "custom_vel_penalty": CUSTOM_VEL_PENALTY,
    # SafeRL and Space Controls both normalize observations, so ALL/EXACT force this on.
    "normalize_obs": NORMALIZE_OBS or SAFERL_ALL or SAFERL_EXACT or SPACE_CONTROLS_ALL or SPACE_CONTROLS_EXACT,
    "paper_time_penalty": PAPER_TIME_PENALTY,
    "paper_unsafe_reward": PAPER_UNSAFE_REWARD,
}


def curriculum_learn(model_id: int, run_dir: str, metadata: dict, seed: int = SEED):
    """Train a drift-assisted docking agent through a series of curriculum stages.

    Each stage trains in a loop until the dock rate hits the stage threshold
    or MAX_EPOCHS_PER_STAGE is reached, then moves to the next stage starting
    from the last saved checkpoint.

    Args:
        model_id: Number used in checkpoint filenames to identify this run.
        run_dir: Folder where checkpoints are saved.
        metadata: The run's metadata dict, already saved once by the
            caller. Mutated in place (metadata["results"]) and re-saved
            once per epoch here, so run_metadata.json stays a near-current
            backup of epoch_history throughout the run instead of only a
            start/end snapshot. Mirrors nodrift_train.py's per-epoch save.
        seed: Optional seed for reproducibility.

    Returns:
        A dict with total_timesteps, total_time_sec, stages_run,
        stage_summary, and epoch_history, for merging into the run's
        metadata file after training completes.
    """
    # N_TORCH_THREADS limits this process; cap_blas_threads_per_worker()
    # limits each worker's BLAS pool. Called here so an import does not
    # apply the limit.
    torch.set_num_threads(N_TORCH_THREADS)
    cap_blas_threads_per_worker()

    # Resume from RESUME_FROM for the first stage only; after that,
    # last_save_path takes over.
    last_save_path = RESUME_FROM if START_STAGE > 0 else None
    total_timesteps = 0
    run_start = time.time()
    stage_summary = []  # one entry per stage: how many epochs/models it took
    epoch_history = []  # one entry per epoch, for the training-curve plot

    # Written one row at a time so an interrupted or stalled run can
    # still be diagnosed from disk.
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
    epoch_log_file = open(epoch_log_path, "w", newline="")
    epoch_log_writer = csv.writer(epoch_log_file)
    epoch_log_writer.writerow(["stage", "epoch", "timesteps", "dock_rate", "fuel_l1", "reward"])
    epoch_log_file.flush()

    for curr in range(START_STAGE, NUM_STAGES):
        print(f'Starting stage {curr}')
        stage_start = time.time()
        stage_timesteps = 0

        configs, threshold = get_curriculum(curr)
        # Apply the REPLACE-style flag values so this matches what
        # resolved_curriculum records below (see apply_flag_config_overrides).
        configs = apply_flag_config_overrides(configs, ENV_FLAGS, agent_mode="drift")
        if SPARSE_REWARD:
            # Override the curriculum's "dense" for this run.
            configs = {**configs, "reward_structure": "sparse"}
        noise = TRAIN_ACTION_NOISE_FRACTION if TRAIN_ACTION_NOISE else 0.0
        env = make_vec_env(
            lambda: wrap_with_action_noise(DriftTrainEnv(**ENV_FLAGS, **configs),
                                           noise, seed),
            n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

        try:
            if curr == 0:
                # Build a new model from scratch for the first stage.
                model = PPO('MlpPolicy', env,
                            learning_rate=LEARNING_RATE,
                            ent_coef=ENT_COEF,
                            gamma=GAMMA,
                            gae_lambda=GAE_LAMBDA,
                            n_steps=512,
                            batch_size=64,
                            target_kl=TARGET_KL,
                            verbose=1,
                            seed=seed,
                            tensorboard_log=os.path.join(run_dir, "tensorboard_logs"))
            else:
                # Load the last checkpoint from the previous stage and
                # keep training in the new, harder environment.
                model = PPO.load(last_save_path, env=env)
                if RESET_STD_ON_STAGE_ADVANCE:
                    with torch.no_grad():
                        model.policy.log_std.fill_(RESET_STD_LOG_VALUE)
                    reset_std = float(np.exp(RESET_STD_LOG_VALUE))
                    print(f"  Reset policy log_std to {RESET_STD_LOG_VALUE} "
                          f"(std {reset_std:.2f}) for stage {curr}.")

            score = 0
            epoch = -1
            prev_save_path = None  # Tracks the previous epoch's checkpoint for cleanup
            best_score = -1.0      # Best dock rate seen this stage
            best_save_path = None  # Checkpoint that produced best_score

            while score < threshold and epoch < MAX_EPOCHS_PER_STAGE - 1:
                epoch += 1
                # TODO: what happens if model diverges?
                model.learn(total_timesteps=TIMESTEPS_PER_EPOCH, tb_log_name=f"stage{curr}",
                            reset_num_timesteps=TENSORBOARD_RESET_NUM_TIMESTEPS)
                stage_timesteps += TIMESTEPS_PER_EPOCH
                total_timesteps += TIMESTEPS_PER_EPOCH

                save_path = os.path.join(
                    run_dir,
                    f"{CHECKPOINT_PREFIX}_{model_id}_{curr}_{epoch}"
                )
                model.save(save_path)
                last_save_path = save_path
                score, mean_fuel, mean_reward = test_model(
                    save_path, curr, seed=EVAL_SEED, env_flags=ENV_FLAGS, return_fuel=True)

                save_name = os.path.basename(save_path)
                print(f"  Epoch {epoch}: dock rate {score:.3f} ({100*score:.1f}%)  |  {save_name}")

                # One row per epoch, for plot_training_curves.py. The same
                # dict feeds epoch_log.csv and run_metadata.json's
                # epoch_history, so the two cannot disagree.
                epoch_row = {
                    "stage": curr,
                    "epoch": epoch,
                    "timesteps": total_timesteps,
                    "dock_rate": score,
                    "fuel_l1": mean_fuel,
                    "reward": mean_reward,
                }
                epoch_history.append(epoch_row)
                epoch_log_writer.writerow([epoch_row[k] for k in (
                    "stage", "epoch", "timesteps", "dock_rate", "fuel_l1", "reward",
                )])
                epoch_log_file.flush()

                # Re-save every epoch so run_metadata.json stays current.
                # "still_running" marks an in-progress run; the final save
                # overwrites it with False and the full results.
                metadata["results"] = {
                    "still_running": True,
                    "total_timesteps": total_timesteps,
                    "stage_summary": list(stage_summary),
                    "epoch_history": epoch_history,
                }
                save_run_metadata(run_dir, metadata)

                if score > best_score:
                    best_score = score
                    best_save_path = save_path

                # If SAVE_ALL_EPOCHS is off, delete the previous epoch's checkpoint
                # now that we have a newer one, but never delete the stage's best.
                if (not SAVE_ALL_EPOCHS and prev_save_path is not None
                        and prev_save_path != best_save_path):
                    zip_path = prev_save_path + ".zip"
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

                prev_save_path = save_path

            if score < threshold:
                is_last_stage = (curr == NUM_STAGES - 1)
                status = "Training complete for final stage" if is_last_stage else "Moving on to next stage"
                print(f"  Stage {curr} hit epoch limit ({MAX_EPOCHS_PER_STAGE}) "
                    f"with dock rate {score:.3f} ({100*score:.1f}%). {status}.")
                # Docking success rate often drops near the end of training (the model can
                # get worse). When a stage reaches its epoch limit, hand the next stage 
                # the best policy this stage produced instead of whatever the policy 
                # drifted to by the last epoch.
                if best_save_path is not None and best_save_path != last_save_path:
                    print(f"  Carrying forward best epoch "
                          f"({os.path.basename(best_save_path)}, dock rate "
                          f"{best_score:.3f}) instead of the final epoch.")
                    last_save_path = best_save_path

            # epoch is 0-indexed, and the loop above stops as soon as score
            # clears threshold, so this is the epoch that passed, or the
            # last one tried if the stage never passed.
            stage_summary.append({
                "stage": curr,
                "epochs_run": epoch + 1,
                "epochs_to_pass": epoch if score >= threshold else None,
                "passed": bool(score >= threshold),
                "best_dock_rate": best_score,
            })

            stage_elapsed = time.time() - stage_start
            stage_min = int(stage_elapsed // 60)
            stage_sec = int(stage_elapsed % 60)
            stage_hrs = stage_elapsed / 3600
            print(f"Stage {curr} complete: {stage_timesteps:,} timesteps, "
                f"{stage_min} min {stage_sec} sec ({stage_hrs:.2f} hrs)")
        finally:
            # Always close subprocess environments, even if training or testing fails.
            env.close()

    epoch_log_file.close()

    # Saved as "final_stage_model", not "final_model": the deployable
    # agent is the whole chain, not any one stage's network on its own.
    # Use resolve_drift_model_chain() to load the chain.
    if last_save_path is not None:
        PPO.load(last_save_path).save(
            os.path.join(run_dir, f"final_stage_model_safe_{model_id}"))

    total_elapsed = time.time() - run_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)
    total_hrs = total_elapsed / 3600
    stages_run = curr - START_STAGE + 1
    avg_elapsed = total_elapsed / max(1, stages_run)
    avg_min = int(avg_elapsed // 60)
    avg_sec = int(avg_elapsed % 60)
    _final_stage_summary = stage_summary[-1]
    _final_stage_note = ("" if _final_stage_summary["passed"]
                          else "  (did not clear this stage's threshold)")
    print(f"\nTraining complete")
    print(f"  Seed:            {seed}")
    print(f"  Final stage:     {_final_stage_summary['stage']}  "
          f"dock rate {_final_stage_summary['best_dock_rate']:.3f} "
          f"({100 * _final_stage_summary['best_dock_rate']:.1f}%)"
          f"{_final_stage_note}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Total time:      {total_min} min {total_sec} sec ({total_hrs:.2f} hrs)"
          f"  ({total_timesteps / max(total_elapsed, 1e-9):,.0f} steps/sec)")
    print(f"  Avg per stage:   {avg_min} min {avg_sec} sec")
    # No single stage matches SafeRL's task (range + precision together),
    # so the whole cascade's timesteps is the comparable total, not any
    # one stage's own dock rate.
    print(f"  Total cascade training cost: {total_timesteps:,} timesteps "
          f"across all {stages_run} stages")
    print(f"  (SafeRL 2022: 660,000 timesteps for their 3D docking task.)")
    print(f"  Comparable as total interaction. Run drift_full_test.py to verify"
          f" the full chain clears SafeRL's 80% win-rate bar.")
    print(f"\nSaved to:        {run_dir}")
    print(f"\nPer-epoch log:   {epoch_log_path}")

    return {
        # False marks a finished run; the per-epoch saves above use True.
        "still_running": False,
        "total_timesteps": total_timesteps,
        "total_time_sec": total_elapsed,
        "total_time_human": format_duration(total_elapsed),
        "stages_run": stages_run,
        # How many epochs (= models) each stage took, and which epoch
        # first cleared that stage's threshold.
        "stage_summary": stage_summary,
        # Per-epoch dock rate across the whole run, the series behind
        # plot_training_curves.py. Kept separate from stage_summary,
        # which only carries each stage's endpoints.
        "epoch_history": epoch_history,
    }


def test_model(path: str, curriculum: int, seed: int = EVAL_SEED,
               env_flags: dict = None, return_fuel: bool = False):
    """Evaluate a checkpoint over 1000 episodes and return the dock rate.

    Called automatically during training after each save. Can also be run
    standalone from __main__ to evaluate a specific checkpoint.

    An episode counts as a success if the agent docks directly or reaches
    a state where drifting to the dock is possible (drift success).

    Args:
        path: Path to the saved model checkpoint.
        curriculum: Stage number, used to set up the right environment.
        seed: Seed for the evaluation episodes. Defaults to EVAL_SEED,
            not the run's training seed, so the same checkpoint always
            scores against the same episodes.
        env_flags: The environment flags to build the eval env with (see
            ENV_FLAGS above; spans SafeRL, custom, and Space Controls
            flags). Must match whatever the checkpoint was
            trained with. None (default) falls back to drift_env.py's
            module defaults.
        return_fuel: If True, also return mean fuel (Delta-v, L1) and
            mean reward across episodes, for logging per-epoch fuel.

    Returns:
        Fraction of episodes that ended in a successful dock or drift,
        or a (dock_rate, mean_fuel_l1, mean_reward) tuple when
        return_fuel is True.
    """
    # test_model runs one episode at a time with a specific seed offset per
    # episode, so it stays a single environment.
    configs, _ = get_curriculum(curriculum)
    env = DriftTrainEnv(**(env_flags or {}), **configs)
    model = PPO.load(path, env=env)
    all_rews, all_fuels, all_fuels_l1, all_docks = [], [], [], []

    for i in range(TEST_EPISODES):
        done = False
        # Offset by episode index so episodes stay varied while the whole
        # test set is reproducible when seed is fixed.
        episode_seed = None if seed is None else seed + i
        obs, info = env.reset(seed=episode_seed)
        epi_fuel, epi_fuel_l1, epi_reward = 0, 0, 0
        is_drift = False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, term, trunc, info = env.step(action)
            epi_reward += reward
            # epi_fuel_l1 (per-axis sum) is the primary fuel metric.
            # epi_fuel (vector norm) is secondary.
            epi_fuel += np.linalg.norm(action) / env.env.m * env.env.step_len
            epi_fuel_l1 += np.sum(np.abs(action)) / env.env.m * env.env.step_len
            # env.step() already ran the coast lookahead and recorded the
            # result; recomputing it here would double the cost of this loop.
            is_drift = info.get("coast_win", False)
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
                all_fuels_l1.append(epi_fuel_l1)

    print(f"  Model stats for stage {curriculum}:")
    print(f"    Dock rate:    {np.mean(all_docks):.3f} ({100*np.mean(all_docks):.1f}%)")
    print(f"    Mean reward:  {np.mean(all_rews):.5f}")
    print(f"    Delta-v, per-axis sum (L1):  mean {np.mean(all_fuels_l1):.3f} m/s  "
          f"median {np.median(all_fuels_l1):.3f} m/s")
    print(f"    Delta-v, vector norm (L2):   mean {np.mean(all_fuels):.3f} m/s  "
          f"median {np.median(all_fuels):.3f} m/s")
    if return_fuel:
        return float(np.mean(all_docks)), float(np.mean(all_fuels_l1)), float(np.mean(all_rews))
    return float(np.mean(all_docks))


def get_curriculum(curriculum: int) -> tt.Tuple[dict, float]:
    """Return environment settings and a pass threshold for a given stage.

    Trained stages are 0 through 8, easy to hard. Coasting is live from
    stage 0 onward; stage 4 only lengthens the lookahead, it doesn't
    turn coasting on. Stage 9 is the evaluation-only config, not trained.

    Args:
        curriculum: Stage number: 0 through 8 for training, or 9 for
            the evaluation config.

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

    # Stage 4: drift_step_len raised to 10 (coasting itself is already
    # live from stage 0; see get_curriculum()'s docstring).
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

    # Stage 9: evaluation config, not a training stage. Full mission range,
    # drift_step_len back to 1 for step-accurate evaluation.
    if curriculum >= 9:
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

    # Resolve to a concrete seed even when SEED is None, so this run can
    # be reproduced later.
    run_seed = SEED if SEED is not None else int(np.random.randint(0, 1_000_000))

    print()  # blank line, so the run's output starts clear of the shell's echoed command
    print(f"Saving models to: {run_dir}\n")
    print(f"Model ID: {model_id}")
    if SEED is not None:
        print(f"Seed: {run_seed}")
    else:
        print(f"Seed: {run_seed} (randomly chosen; set SEED to this value to reproduce)")
    if START_STAGE >= NUM_STAGES:
        raise ValueError(
            f"START_STAGE is {START_STAGE} but NUM_STAGES is {NUM_STAGES}, so "
            f"there is no stage left to train. Set START_STAGE below NUM_STAGES."
        )
    if START_STAGE > 0 and RESUME_FROM is None:
        # Caught here so the error is clear, instead of a confusing
        # failure deep inside PPO.load(None).
        raise ValueError(
            f"START_STAGE is {START_STAGE} but RESUME_FROM is None. "
            f"Starting mid-curriculum needs a checkpoint to resume from; "
            f"set RESUME_FROM to the .zip path to continue from, or set "
            f"START_STAGE = 0 to train from scratch."
        )
    if RESUME_FROM is not None:
        print(f"Resuming from: {RESUME_FROM} (starting at stage {START_STAGE})")
    print(f"CPU: n_envs={N_ENVS}  torch_threads={N_TORCH_THREADS}")
    print(f"Eval seed: {EVAL_SEED} (fixed, so stage pass/fail is comparable across runs)")

    # All default False; True means a deliberate ablation.
    print("Env/reward options:")
    print("  SafeRL (2022):")
    print(f"    {'Observation (speed + max_vel_limit):':<38} {ENV_FLAGS['saferl_obs']}")
    print(f"    {'Exponential distance reward:':<38} {ENV_FLAGS['saferl_exp_dist_reward']}")
    print(f"    {'Delta-v fuel penalty:':<38} {ENV_FLAGS['saferl_delta_v_penalty']}")
    print(f"    {'Success time bonus:':<38} {ENV_FLAGS['saferl_success_time_bonus']}")
    print(f"    {'Success reward = +2:':<38} {ENV_FLAGS['saferl_success_reward']}")
    print(f"    {'Budgeted velocity constraint:':<38} {ENV_FLAGS['saferl_vel_constraint']}")
    print(f"    {'40km boundary:':<38} {ENV_FLAGS['saferl_max_distance']}")
    print(f"    {'No time penalty:':<38} {ENV_FLAGS['saferl_no_time_penalty']}")
    print("  Space Controls (2024):")
    print(f"    {'Observation (drop timestep):':<38} {ENV_FLAGS['space_controls_obs']}")
    print(f"    {'Flat time penalty:':<38} {ENV_FLAGS['space_controls_time_penalty']}")
    print(f"    {'80% init speed limit:':<38} {ENV_FLAGS['space_controls_init_vel_fraction']}")
    print(f"    {'Speed limit from dock radius:':<38} {ENV_FLAGS['space_controls_speed_limit_offset']}")
    print(f"    {'Velocity constraint (no budget):':<38} {ENV_FLAGS['space_controls_vel_constraint']}")
    print(f"    {'10m dock radius:':<38} {ENV_FLAGS['space_controls_dock_radius']}")
    print(f"    {'800m boundary:':<38} {ENV_FLAGS['space_controls_max_distance']}")
    print(f"    {'0.1N low thrust:':<38} {ENV_FLAGS['space_controls_low_thrust']}")
    print("  Custom:")
    print(f"    {'Braking margin observation:':<38} {ENV_FLAGS['custom_braking_margin_obs']}")
    print(f"    {'Proximity-scaled velocity penalty:':<38} {ENV_FLAGS['custom_vel_penalty']}")
    print(f"    {'Observation normalization:':<38} {ENV_FLAGS['normalize_obs']}")
    print("  Paper (this project's Methods text):")
    print(f"    {'Flat -0.0005 time penalty:':<38} {ENV_FLAGS['paper_time_penalty']}")
    print(f"    {'Unsafe-termination reward = 0:':<38} {ENV_FLAGS['paper_unsafe_reward']}")
    _any_space_controls_on = any(
        ENV_FLAGS[k] for k in ENV_FLAGS if k.startswith("space_controls_"))
    if _any_space_controls_on and not (
            ENV_FLAGS["saferl_exp_dist_reward"] and ENV_FLAGS["saferl_delta_v_penalty"]):
        print("  WARNING: Space Controls is on but "
              "SAFERL_EXP_DIST_REWARD/SAFERL_DELTA_V_PENALTY is not; "
              "this run has neither paper's reward function.")
    print("Training-side ablations (outside env/reward options above):")
    print(f"  {'Sparse reward:':<38} {SPARSE_REWARD}")
    print(f"  {'Train-time action noise:':<38} {TRAIN_ACTION_NOISE}"
          f"{f' (+/-{TRAIN_ACTION_NOISE_FRACTION * 100:.0f}%)' if TRAIN_ACTION_NOISE else ''}")
    print(f"  {'SafeRL gamma/GAE lambda:':<38} {SAFERL_GAMMA_LAMBDA}"
          f" (gamma={GAMMA}, gae_lambda={GAE_LAMBDA})\n")

    # Save the curriculum and reward settings this run trains under, so
    # the checkpoint stays reproducible even if this file changes later.
    _probe_env = DriftDockingEnv3D()
    def _frozen_stage(stage):
        """One stage's config and threshold, with the same flag overrides
        the training loop applies, so this records what trained.
        """
        stage_config, stage_threshold = get_curriculum(stage)
        stage_config = apply_flag_config_overrides(stage_config, ENV_FLAGS, agent_mode="drift")
        # fixed_start is always False during training, so fixed_state is
        # unused; drop it rather than record a dead default.
        stage_config = {k: v for k, v in stage_config.items() if k != "fixed_state"}
        return stage_config, stage_threshold

    resolved_curriculum = {}
    for stage in range(NUM_STAGES):
        stage_config, stage_threshold = _frozen_stage(stage)
        resolved_curriculum[stage] = {"config": stage_config, "threshold": stage_threshold}
    # Key order = JSON key order: identity/provenance fields, then
    # flags/hyperparameters, then resolved_curriculum last since its dump is long.
    metadata = {
        "run_name": RUN_NAME,
        "run_folder": os.path.basename(run_dir),
        "checkpoint_prefix": CHECKPOINT_PREFIX,
        "model_id": model_id,
        "seed": run_seed,
        "seed_was_fixed": SEED is not None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "package_versions": get_package_versions(),
        "start_stage": START_STAGE,
        "num_stages": NUM_STAGES,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "timesteps_per_epoch": TIMESTEPS_PER_EPOCH,
        "n_envs": N_ENVS,
        "eval_seed": EVAL_SEED,
        "test_episodes": TEST_EPISODES,
        "ent_coef": ENT_COEF,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "learning_rate": LEARNING_RATE,
        "target_kl": TARGET_KL,
        "env_flags": group_env_flags(ENV_FLAGS),
        "reward_structure": "sparse" if SPARSE_REWARD else "dense",
        "reward_coefficients": {
            "dist_coeff": _probe_env.dist_coeff,
            "time_penalty": _probe_env.time_penalty,
        },
        "train_action_noise": TRAIN_ACTION_NOISE,
        "train_action_noise_fraction": (
            TRAIN_ACTION_NOISE_FRACTION if TRAIN_ACTION_NOISE else 0.0),
        "saferl_gamma_lambda": SAFERL_GAMMA_LAMBDA,
        "reset_std_on_stage_advance": RESET_STD_ON_STAGE_ADVANCE,
        "reset_std_log_value": RESET_STD_LOG_VALUE,
        "resolved_curriculum": resolved_curriculum,
    }
    metadata_path = save_run_metadata(run_dir, metadata)
    print(f"Saving run metadata to: {metadata_path}\n")

    results = curriculum_learn(model_id, run_dir, metadata, seed=run_seed)

    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["results"] = results
    # Put the long curriculum details last so the run results are easy to find.
    metadata["resolved_curriculum"] = metadata.pop("resolved_curriculum")
    metadata_path = save_run_metadata(run_dir, metadata)
    print(f"\nRun metadata saved to: {metadata_path}")

    # To test a specific checkpoint after training:
    # test_model('data/checkpoints/safe_PPO_6/safe_ppo_model_6_7_1.zip', 7)
