"""Evaluate a full curriculum as one continuous episode.

1. Start at the hardest checkpoint.
2. Move to the next easier checkpoint after docking.
3. Use only the final stage to determine the result.

- Drift chains use the shared rollout helper.
- Nodrift chains reset docking settings and counters before each model switch.
- Set AGENT_MODE to choose drift or nodrift mode.
- Print results or save them to saved_data/curriculum_evaluation/.
"""

import os
import contextlib
import io
import time
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
import drift_env
import docking_env
from evaluation_utilities import (
    extract_stage,
    resolve_checkpoint_paths,
    resolve_agent_mode,
    resolve_obs_flags,
    resolve_env_flags_from_metadata,
    resolve_env_config_best_effort,
    resolve_drift_stage_thresholds,
    rollout_drift_chain_episode,
    classify_failure_nodrift,
    interquartile_mean,
    write_tidy_csv,
    SPACE_CONTROLS_FLAG_KEYS,
    NODRIFT_REWARD_FLAG_KEYS,
    DRIFT_REWARD_FLAG_KEYS,
    MAX_STEPS_PER_EPISODE,
)

# "drift": DriftTestEnv. "nodrift": SpaceCraftDockingEnv3D.
# "auto": detect from checkpoint filename ("nodrift" in name -> nodrift).
AGENT_MODE = "auto"

# --- Settings ---
# Grouped by what they affect; execution starts below.

# Checkpoints:

# Which checkpoint chain to evaluate.
#   "latest"                       newest run folder
#   "run:<folder>" or a folder     newest .zip in that folder
#   "pattern:<glob>"               custom glob
#   an exact .zip path             that one checkpoint
CHECKPOINT = "latest"

# Only used when CHECKPOINT is "latest": restricts which run folders count,
# so "latest" tracks the newest drift run (safe_PPO_* naming).
LATEST_FOLDER_PREFIX = "safe_PPO"

# Max curriculum-stage checkpoints per episode. Set high (e.g. 100) so
# resolve_checkpoint_paths() can limit it to the available checkpoints.
NUM_MODELS = 100

# Run setup:

# Fixed seed, so test runs are repeatable.
SEED = 0
np.random.seed(SEED)

# Number of evaluation episodes to run.
NUM_EVAL_EPISODES = 10

# True:  exact closed-form CWH propagation instead of solve_ivp. Matches
#        what the trainers use, and roughly 8x faster; agrees with
#        solve_ivp to about 1e-11 m, far below any dock threshold.
# False: solve_ivp (RK45), the same integrator the environments used to
#        default to. Use this to check a result against the original solver.
FAST_ANALYTIC_PROPAGATION = True
drift_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION
docking_env.FAST_ANALYTIC_PROPAGATION = FAST_ANALYTIC_PROPAGATION

# Console output:

# Position and velocity norms at each step and every stage switch.
# True:  print all of it.
# False: only per-stage results and the final summary.
VERBOSE = False

# File output:

# True: save one CSV row per episode (win, fuel, episode length, elapsed
# time, failure type) to saved_data/curriculum_evaluation/, timestamped
# so repeated runs don't overwrite each other.
EXPORT_RESULTS_CSV = True

# Root folder that contains saved model checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")


def _resolve_stage_thresholds(checkpoint_path: str, live_get_curriculum) -> dict:
    """Resolve one checkpoint's docking thresholds, metadata first.

    Uses resolve_env_config_best_effort() instead of a raw
    get_curriculum(stage) call, which can return the wrong
    thresholds if the curriculum has since been renumbered. The live
    get_curriculum(stage) result is still passed in as a fallback for
    checkpoints with no run_metadata.json.

    Args:
        checkpoint_path: Full path to the checkpoint needing thresholds.
        live_get_curriculum: The agent mode's own get_curriculum
            function, used only to build the fallback config.

    Returns:
        The resolved config dict, with at least 'pos_thresh' and
        'speed_thresh'.
    """
    stage = extract_stage(checkpoint_path)
    fallback_config, _ = live_get_curriculum(stage)
    config, source, verified = resolve_env_config_best_effort(
        checkpoint_path, fallback_config=fallback_config
    )
    if not verified:
        print(f"    Env config: {source}")
    return config


def make_env(mode: str):
    """Create the evaluation environment and curriculum function for the given mode.

    Args:
        mode: "drift" for DriftTestEnv, "nodrift" for SpaceCraftDockingEnv3D.

    Returns:
        A tuple (env_constructor, is_docked_fn, is_drifting_fn,
        apply_stage_fn, cap_timestep_fn). apply_stage_fn sets a
        checkpoint's own docking thresholds and resets its timestep and
        fuel_used.

    Raises:
        ValueError: If mode is not "drift" or "nodrift".
    """
    if mode == "drift":
        from drift_env import DriftTestEnv
        from drift_initial_trainer import get_curriculum
        from evaluation_utilities import resolve_drift_eval_config

        def build_env(saferl_obs=False, custom_braking_margin_obs=False,
                      **extra_flags):
            """Build a DriftTestEnv on the shared drift eval config."""
            # apply_stage() applies stage thresholds later. verbose=False
            # since __main__ prints the config once.
            curriculum, _ = resolve_drift_eval_config(verbose=False)
            return DriftTestEnv(
                saferl_obs=saferl_obs,
                custom_braking_margin_obs=custom_braking_margin_obs,
                **extra_flags,
                **curriculum,
            )

        def is_docked(env):
            """True when the wrapped env reports a successful dock."""
            return env.env.is_docked()

        def is_drifting(env):
            """True while the coast hold is active."""
            return env.is_drifting

        def apply_stage(env, checkpoint_path):
            """Point the env at one checkpoint's own docking thresholds."""
            stage_config = _resolve_stage_thresholds(checkpoint_path, get_curriculum)
            env.env.dock_dist = stage_config['pos_thresh']
            env.env.dock_speed = stage_config['speed_thresh']
            env.env.state[6] = 0
            env.env.fuel_used = 0

        def cap_timestep(env):
            """No-op: apply_stage() already resets state[6] at each handoff."""
            pass

        return build_env, is_docked, is_drifting, apply_stage, cap_timestep

    elif mode == "nodrift":
        from docking_env import SpaceCraftDockingEnv3D
        from nodrift_initial_trainer import get_curriculum, NUM_STAGES

        def build_env(saferl_obs=False, custom_braking_margin_obs=False,
                      **extra_flags):
            """Build a SpaceCraftDockingEnv3D on the hardest stage's config."""
            # apply_stage() handles per-checkpoint thresholds separately.
            curriculum, _ = get_curriculum(NUM_STAGES - 1)
            # Override to full range for the evaluation episode.
            curriculum['max_episode_len'] = 5_000
            curriculum['min_init_pos_bound'] = 75
            curriculum['max_init_pos_bound'] = 150
            return SpaceCraftDockingEnv3D(
                saferl_obs=saferl_obs,
                custom_braking_margin_obs=custom_braking_margin_obs,
                **extra_flags,
                **curriculum,
            )

        def is_docked(env):
            """True when the env reports a successful dock."""
            return env.is_docked()

        def is_drifting(env):
            """Always False: the nodrift agent has no coast mechanic."""
            return False

        def apply_stage(env, checkpoint_path):
            """Point the env at one checkpoint's own docking thresholds."""
            stage_config = _resolve_stage_thresholds(checkpoint_path, get_curriculum)
            env.dock_dist = stage_config['pos_thresh']
            env.dock_speed = stage_config['speed_thresh']
            env.state[6] = 0
            env.fuel_used = 0
            env.vel_violation_reward_sum = 0  # New constraint budget per stage

        def cap_timestep(env):
            """No-op: nodrift's long eval episodes do not need a mid-stage limit."""
            pass

        return build_env, is_docked, is_drifting, apply_stage, cap_timestep

    else:
        raise ValueError(f"Unknown AGENT_MODE: {mode!r}. Use 'drift' or 'nodrift'.")


def load_and_validate_models(checkpoint_files, checkpoint_spec):
    """Load trained models from checkpoint files and prepare them for evaluation.

    Sorts by curriculum stage (lowest to highest) and loads each model on CPU.

    Args:
        checkpoint_files: List of file paths to model checkpoint files.
        checkpoint_spec: The original specification (used only for error messages).

    Returns:
        A tuple of loaded PPO model objects, sorted by stage.

    Raises:
        FileNotFoundError: If no checkpoint files were provided.
    """
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoints found for spec: {checkpoint_spec}")

    checkpoint_files = sorted(checkpoint_files, key=extract_stage)

    # Use CPU: these small models are generally faster here than on GPU.
    models = []
    for p in checkpoint_files:
        models.append(PPO.load(p, device="cpu"))

    return tuple(models)


# Resolve checkpoint files and validate them before loading any models.
if __name__ == "__main__":
    checkpoint_files = resolve_checkpoint_paths(
        CHECKPOINT, CHECKPOINT_ROOT, num_models=NUM_MODELS, folder_prefix=LATEST_FOLDER_PREFIX
    )

    # NUM_MODELS above is only a limit; use however many checkpoints were found.
    NUM_MODELS = len(checkpoint_files)

    stages_found = [extract_stage(p) for p in checkpoint_files]

    if -1 in stages_found:
        raise ValueError(
            f"Could not parse stage number from one or more checkpoint filenames: {checkpoint_files}"
        )
    if len(set(stages_found)) != len(stages_found):
        raise ValueError(
            f"Duplicate stage numbers found in selected checkpoints: {stages_found}"
        )
    max_stage = max(stages_found)
    if len(checkpoint_files) > max_stage + 1:
        raise ValueError(
            f"Requested {len(checkpoint_files)} models but only stages 0 to {max_stage} exist."
        )
    if len(checkpoint_files) < 1:
        raise FileNotFoundError(f"No checkpoints found for spec: {CHECKPOINT}")

    AGENT_MODE = resolve_agent_mode(AGENT_MODE, checkpoint_files[-1])

    print(f"Agent mode: {AGENT_MODE}")
    print("Evaluating models:")
    for i, p in enumerate(checkpoint_files):
        print(f"  Model {i+1}: {p}")

    models = load_and_validate_models(checkpoint_files, CHECKPOINT)
    build_env, is_docked_fn, is_drifting_fn, apply_stage_fn, cap_timestep_fn = make_env(AGENT_MODE)

    if AGENT_MODE == "drift":
        # Print the drift config once; build_env() repeats it every episode. 
        # Stage 9 is not saved in checkpoint metadata.
        from evaluation_utilities import resolve_drift_eval_config
        resolve_drift_eval_config(verbose=True)

    # Match model inputs to the environment's observation layout.
    space_controls_flags = resolve_env_flags_from_metadata(checkpoint_files[0], keys=SPACE_CONTROLS_FLAG_KEYS)
    saferl_obs, custom_braking_margin_obs = resolve_obs_flags(
        models[0].observation_space.shape[0],
        space_controls_obs=space_controls_flags.get("space_controls_obs", False),
    )
    # Reward settings are read from metadata; they affect scoring but not input size.
    _reward_keys = DRIFT_REWARD_FLAG_KEYS if AGENT_MODE == "drift" else NODRIFT_REWARD_FLAG_KEYS
    reward_flags = resolve_env_flags_from_metadata(checkpoint_files[0], keys=_reward_keys)

    fuels = []
    fuels_l1 = []
    wins = []
    episode_lengths = []
    elapsed_seconds_list = []
    outcomes = []
    failure_types = []

    if AGENT_MODE == "drift":
        # Reorder models for the drift rollout: the helper expects the
        # hardest stage first, while saved checkpoints are kept easiest-first.
        drift_models = list(reversed(models))
        drift_model_paths = list(reversed(checkpoint_files))
        drift_thresholds = [
            resolve_drift_stage_thresholds(str(p), verbose=False)
            for p in drift_model_paths
        ]

    # Every episode starts on the hardest stage (last model) and works down to model 1.
    model_start = NUM_MODELS

    for i in range(NUM_EVAL_EPISODES):
        episode_start = time.time()
        np.random.seed(SEED + i)
        print(f"Episode {i+1}/{NUM_EVAL_EPISODES}")

        env = build_env(saferl_obs, custom_braking_margin_obs, **space_controls_flags, **reward_flags)
        obs, info = env.reset(seed=SEED + i)

        if AGENT_MODE == "drift":
            ep_result = rollout_drift_chain_episode(
                env, obs, drift_models, drift_thresholds, drift_model_paths,
                max_steps=MAX_STEPS_PER_EPISODE,
                verbose=VERBOSE,
                print_stage_results=True,
            )
            wins.append(1 if ep_result["win"] else 0)
            fuels.append(ep_result["fuel"])
            fuels_l1.append(ep_result["fuel_l1"])
            episode_lengths.append(ep_result["episode_length"])
            elapsed_seconds_list.append(ep_result["elapsed_seconds"])
            outcomes.append(ep_result["outcome"])
            failure_types.append(ep_result["failure_type"])
            episode_time = time.time() - episode_start
            print(
                f"  Episode result: {'WIN' if ep_result['win'] else 'LOSS'}  "
                f"Delta-v L1: {ep_result['fuel_l1']:.2f}  L2: {ep_result['fuel']:.2f}  "
                f"Eval time: {episode_time:.2f}s"
            )
            continue

        # No shared helper for this mode; keep rollout manual.
        model_num = model_start
        epi_fuel = 0
        epi_fuel_l1 = 0
        done = False

        # Match stage thresholds to the current model.
        apply_stage_fn(env, checkpoint_files[model_num - 1])

        if VERBOSE:
            print(f"  Starting with model {model_num}")

        step_count = 0
        while not done:
            cap_timestep_fn(env)

            if is_drifting_fn(env):
                # Drift phase: hold steady with zero thrust.
                action = np.array([0.0, 0.0, 0.0])
            else:
                expected_obs_size = models[model_num - 1].observation_space.shape[0]
                action = models[model_num - 1].predict(
                    obs[:expected_obs_size], deterministic=True
                )[0]

            # epi_fuel_l1 (per-axis sum) is the primary fuel metric; epi_fuel
            # (vector norm) is secondary. Summed manually since apply_stage_fn
            # resets env.fuel_used on each model switch.
            if hasattr(env, "env") and hasattr(env.env, "m"):
                mass, step_len = env.env.m, env.env.step_len
            elif hasattr(env, "m"):
                mass, step_len = env.m, env.step_len
            else:
                mass, step_len = None, None

            if mass is not None:
                epi_fuel += float(np.linalg.norm(action)) / mass * step_len
                epi_fuel_l1 += float(np.sum(np.abs(action))) / mass * step_len
            else:
                # Fallback: raw action magnitude (not Delta-v)
                epi_fuel += np.linalg.norm(action)
                epi_fuel_l1 += np.sum(np.abs(action))

            # Suppress environment prints (WIN!, UNSAFE!) so only this script's
            # per-stage result lines are shown.
            with contextlib.redirect_stdout(io.StringIO()):
                obs, reward, term, trunc, info = env.step(action)

            if VERBOSE:
                # Read from env state (meters), not the possibly normalized obs.
                state = env.env.state if AGENT_MODE == "drift" else env.state
                r = np.linalg.norm(state[0:3])
                v = np.linalg.norm(state[3:6])
                print(f"  r={r:.3f}, v={v:.3f}")

            done = term or trunc
            step_count += 1

            if step_count >= MAX_STEPS_PER_EPISODE and not done:
                # Hit the safety limit without a real ending. Record it as a
                # loss rather than dropping the episode entirely.
                print(f"  Episode hit the {MAX_STEPS_PER_EPISODE}-step limit "
                      f"without ending; recording it as a loss.")
                wins.append(0)
                fuels.append(epi_fuel)
                fuels_l1.append(epi_fuel_l1)
                episode_lengths.append(step_count)
                elapsed_seconds_list.append(step_count * env.step_len)
                outcomes.append("loss")
                failure_types.append("timeout")
                break

            if done:
                stage_num = extract_stage(checkpoint_files[model_num - 1])
                stage_result = is_docked_fn(env)
                print(f"  Stage {stage_num} / model {model_num}: {'SUCCESS' if stage_result else 'FAIL'}")

                if stage_result and model_num != 1:
                    # This stage docked and the chain is not done: switch to
                    # the next easier model.
                    done = False
                    model_num = max(model_num - 1, 1)
                    # Reset docking thresholds and timestep count for the
                    # active stage so is_docked() uses the correct range.
                    apply_stage_fn(env, checkpoint_files[model_num - 1])
                else:
                    # Either the final stage ended, or a mid-chain stage ended
                    # some other way (crash, out of bounds, unsafe, timeout).
                    # Either way the episode is over.
                    wins.append(int(stage_result))
                    fuels.append(epi_fuel)
                    fuels_l1.append(epi_fuel_l1)
                    episode_lengths.append(step_count)
                    elapsed_seconds_list.append(step_count * env.step_len)
                    outcomes.append("win" if stage_result else "loss")
                    failure_types.append(None if stage_result else classify_failure_nodrift(env))
                    episode_time = time.time() - episode_start
                    print(
                        f"  Episode result: {'WIN' if stage_result else 'LOSS'}  "
                        f"Delta-v L1: {epi_fuel_l1:.2f}  L2: {epi_fuel:.2f}  "
                        f"Eval time: {episode_time:.2f}s"
                    )

    # Final summary.
    print()
    print(f"--- Results ({AGENT_MODE} agent) ---")
    print(f"Wins: {sum(wins)}/{NUM_EVAL_EPISODES} ({100 * np.mean(wins):.1f}%)")

    if len(fuels) > 0:
        # L1 (per-axis sum) is the primary fuel metric. L2 (vector norm) is secondary.
        print()
        print(f"Delta-v, per-axis sum (L1, m/s):")
        print(f"  Mean:    {np.mean(fuels_l1):.3f}")
        print(f"  Median:  {np.median(fuels_l1):.3f}")
        print(f"  IQM:     {interquartile_mean(fuels_l1):.3f}")
        print(f"  Min:     {min(fuels_l1):.3f}")
        print(f"  Max:     {max(fuels_l1):.3f}")
        print(f"  25th pct: {np.percentile(fuels_l1, 25):.3f}")
        print(f"  50th pct: {np.percentile(fuels_l1, 50):.3f}")
        print(f"  75th pct: {np.percentile(fuels_l1, 75):.3f}")

        print()
        print(f"Delta-v, vector norm (L2, m/s):")
        print(f"  Mean:    {np.mean(fuels):.3f}")
        print(f"  Median:  {np.median(fuels):.3f}")
        print(f"  IQM:     {interquartile_mean(fuels):.3f}")
        print(f"  Min:     {min(fuels):.3f}")
        print(f"  Max:     {max(fuels):.3f}")
        print(f"  25th pct: {np.percentile(fuels, 25):.3f}")
        print(f"  50th pct: {np.percentile(fuels, 50):.3f}")
        print(f"  75th pct: {np.percentile(fuels, 75):.3f}")
    else:
        print("  No completed episodes recorded.")

    if EXPORT_RESULTS_CSV and len(wins) > 0:
        # One row per episode, formatted for Seaborn, matching the
        # other eval scripts' CSV shape. Timestamped so a re-run does not
        # overwrite an earlier CSV.
        run_folder_name = os.path.basename(os.path.dirname(checkpoint_files[0]))
        rows = [
            {
                "condition": run_folder_name,
                "agent_mode": AGENT_MODE,
                "episode_idx": i,
                "win": bool(wins[i]),
                "outcome": outcomes[i],
                "failure_type": failure_types[i],
                "fuel_l2": fuels[i],
                "fuel_l1": fuels_l1[i],
                "episode_length": episode_lengths[i],
                "elapsed_seconds": elapsed_seconds_list[i],
            }
            for i in range(len(wins))
        ]
        _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_tidy_csv(
            rows,
            os.path.join(SCRIPT_DIR, "saved_data", "curriculum_evaluation",
                         f"curriculum_evaluation_{run_folder_name}_{_timestamp}.csv"),
            data_label="episode results",
        )

    print()
    print(f"Models evaluated:")
    for i, p in enumerate(checkpoint_files):
        print(f"  [{i+1}] stage={extract_stage(p)}  {p}")
