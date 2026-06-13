import numpy as np

from drift_env import DriftTestEnv
from initial_trainer import get_curriculum
from stable_baselines3 import PPO

import glob
import os

# Use a fixed seed so test runs are repeatable.
SEED = 0
np.random.seed(SEED)

# CHECKPOINT CONFIG

# Options:
# "latest"                                  -> newest .zip across all runs
# "run:safe_PPO_*"                          -> newest .zip inside run folder safe_PPO_*
# "exact:path"                              -> direct file path
# directory:path                            -> load latest .zip files from a run folder
# "pattern:safe_ppo_model_*_1.zip"          -> custom glob inside checkpoint dir
# r"C:\...\checkpoint.zip"                  -> absolute path

# Edit this variable to select which checkpoint(s) to evaluate. See options above.
CHECKPOINT = "latest"

# Root folder that contains saved model checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(SCRIPT_DIR, "data", "checkpoints")

def resolve_checkpoint_paths(checkpoint_spec: str):
    """Find and return model checkpoint files to evaluate.
    
    Takes a description of which checkpoints to use and returns the file paths.
    You can specify checkpoints in several simple ways:

    Args:
        checkpoint_spec: How to find the checkpoints. Use one of these:
            - "latest": Find the 4 newest models from any saved run
            - "run:safe_PPO_1": Find the 4 newest models from the safe_PPO_1 folder
            - "data/checkpoints/safe_PPO_1": Load models from this folder
            - "model.zip": Load a single specific model file
            - "pattern:safe_ppo_model_*_1.zip": Find all files matching this pattern

    Returns:
        A list of file paths to model checkpoint files.
    """

    # Case 1: A direct path was provided, so return it immediately.
    if os.path.isfile(checkpoint_spec):
        return [checkpoint_spec]

    # Case 1b: A directory was provided, so load the newest checkpoint files from it.
    if os.path.isdir(checkpoint_spec):
        files = sorted(glob.glob(os.path.join(checkpoint_spec, "*.zip")))
        if not files:
            raise FileNotFoundError(
                f"No checkpoint .zip files found in directory: {checkpoint_spec}"
            )
        return files[-4:]

    # Case 2: Load the newest checkpoints across every run folder.
    if checkpoint_spec == "latest":
        pattern = os.path.join(CHECKPOINT_ROOT, "**", "*.zip")
        files = glob.glob(pattern, recursive=True)

        if not files:
            raise FileNotFoundError("No checkpoints found")

        # Sort by filename first (more stable than timestamps)
        files = sorted(files)

        return files[-4:]

    # Case 3: Load checkpoints from one specific run folder.
    if checkpoint_spec.startswith("run:"):
        run_name = checkpoint_spec.split("run:")[1]
        pattern = os.path.join(CHECKPOINT_ROOT, run_name, "*.zip")
        files = sorted(glob.glob(pattern))
        # Keep only the latest four checkpoints for that run.
        return files[-4:]

    # Case 4: Use a custom glob pattern under the checkpoint root.
    if checkpoint_spec.startswith("pattern:"):
        pattern = checkpoint_spec.split("pattern:")[1]
        full_pattern = os.path.join(CHECKPOINT_ROOT, "**", pattern)
        files = glob.glob(full_pattern, recursive=True)
        # Sort so the checkpoint order is predictable.
        return sorted(files)[-4:]

    # If none of the supported formats match, stop with an error.
    raise ValueError(f"Invalid checkpoint spec: {checkpoint_spec}")

# Resolve the checkpoint files using the selected strategy.
checkpoint_files = resolve_checkpoint_paths(CHECKPOINT)

# Make sure at least one checkpoint exists before continuing.
if len(checkpoint_files) < 1:
    raise FileNotFoundError(
        f"No checkpoints found for spec: {CHECKPOINT}"
    )

# Display the checkpoint files that will be evaluated.
print("Evaluating models:")

for i, p in enumerate(checkpoint_files):
    print(f"  Model {i+1}: {p}")

def extract_stage(path):
    """Get the curriculum stage number from a model filename.
    
    Model files are named like: safe_ppo_model_3_1.zip
    This function extracts the stage number (the second-to-last number).
    
    For example, "safe_ppo_model_3_1.zip" has stage 3.

    Args:
        path: The file path to a model checkpoint file.

    Returns:
        The curriculum stage number (an integer), or -1 if it cannot be read.
    """
    base = os.path.basename(path).replace(".zip", "")
    parts = base.split("_")

    # safe parsing: safe_ppo_model_{stage}_{run}
    try:
        return int(parts[-2])
    except:
        return -1

def load_and_validate_models(checkpoint_files, checkpoint_spec):
    """Load trained models from checkpoint files and prepare them for evaluation.
    
    This function:
    1. Checks that we have at least 4 model files
    2. Sorts them by curriculum stage (1, 2, 3, 4)
    3. Loads each model into memory
    4. Returns all 4 models ready to use
    
    Args:
        checkpoint_files: List of file paths to model checkpoint files.
        checkpoint_spec: The original specification (used only for error messages).
    
    Returns:
        Four trained model objects: (model_1, model_2, model_3, model_4)
        
    Raises:
        FileNotFoundError: If no checkpoint files were provided.
        ValueError: If fewer than 4 checkpoint files were found.
    """
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoints found for spec: {checkpoint_spec}")
    
    if len(checkpoint_files) < 4:
        raise ValueError(
            f"Need at least 4 checkpoints for evaluation, got {len(checkpoint_files)}: "
            f"{checkpoint_files}"
        )
    
    # Sort by curriculum stage
    checkpoint_files = sorted(checkpoint_files, key=extract_stage)
    
    # Load models
    models = [PPO.load(p) for p in checkpoint_files]
    
    # Always take the last 4 curriculum stages
    models = models[-4:]
    
    return tuple(models)

# Load and validate models
model_1, model_2, model_3, model_4 = load_and_validate_models(checkpoint_files, CHECKPOINT)

# Store fuel usage and success values.
fuels = []
wins = []

# Keep a copy of the results for later reuse if needed.
results = {
    "checkpoints": checkpoint_files,
    "wins": [],
    "fuels": [],
}

# Run a fixed number of evaluation episodes.
num_eval_episodes = 10

for i in range(num_eval_episodes):
    # Re-seed each episode so the test set remains deterministic.
    np.random.seed(SEED + i)

    print(f"Episode {i+1}/{num_eval_episodes}")
    print(f"Starting with model_4")

    # Track fuel usage for this episode.
    epi_fuel = 0

    # Create the test curriculum and instantiate the environment.
    curriculum, _ = get_curriculum(10)
    env = DriftTestEnv(**curriculum)
    obs, info = env.reset(seed=SEED + i)
    done = False

    # Start from the last curriculum stage and work backward if needed.
    model_num = 4
    model = model_4
    t_step = 0

    # Keep stepping until the episode fully completes.
    while not done:
        # During drift, or during selected time windows, apply zero thrust.
        # This makes the evaluation behavior match the intended test policy.
        if env.is_drifting or (t_step % 4) in list(range(5, 10)): # TODO
            # if you took 5 actions, then drift for 5 actions, then repeat. 
            action = np.array([0.0, 0.0, 0.0])
        else:
            # Ask the current model for the next action.
            action = model.predict(obs, deterministic=True)[0]

        # Track total fuel by accumulating the action magnitude.
        epi_fuel += np.linalg.norm(action)

        # Advance the environment one step.
        obs, reward, term, trunc, info = env.step(action)
        t_step += 1

        # Print position and velocity norms for debugging and monitoring.
        print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        done = term or trunc

        # If the episode ended, either record the final result or switch stages.
        if done:
            if model_num == 1:
                # Final stage completed, so record whether docking succeeded.
                success = env.env.is_docked() # env.env is the underlying DriftDockEnv instance
                wins.append(int(success))
                fuels.append(epi_fuel)

                results["wins"].append(int(success))
                results["fuels"].append(epi_fuel)
            else:
                # If docking is not yet finished, prepare to evaluate the next
                # easier curriculum stage by resetting the environment state.
                done = False
                env.is_drifting = False
                t_step = 0
                model_num -= 1

                # Update thresholds to match the next stage in the curriculum.
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

# Print a summary of the test run.
print("\n=== TEST RESULTS ===")
print(f"Wins: {sum(wins)}/{num_eval_episodes} ({100 * np.mean(wins):.1f}%)")
print(f"\nFuel Consumption Statistics:")
print(f"  Mean:   {np.mean(fuels):.2f}")
print(f"  Median: {np.median(fuels):.2f}")
print(f"  Min:    {min(fuels):.2f}")
print(f"  Max:    {max(fuels):.2f}")

# Sort fuel usage values for percentile reporting.
fuels_sorted = np.sort(fuels)
print("\nPercentiles:")
print(f"  25th: {np.percentile(fuels, 25):.2f}")
print(f"  50th: {np.percentile(fuels, 50):.2f}")
print(f"  75th: {np.percentile(fuels, 75):.2f}")
