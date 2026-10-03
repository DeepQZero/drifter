# Low-Fuel Deep Reinforcement Learning Spacecraft Control through Lookahead Drifting

**Drifter** is a reinforcement learning (RL) codebase for low-fuel spacecraft docking. It contains the Drifter-Refine and Drifter-Learn algorithms, a Gymnasium docking environment, Proximal Policy Optimization (PPO) and Linear-Quadratic Regulator (LQR) baselines, and the evaluation code for the submitted paper below:

Marcos Sanson, Nikita Agrawal, Andrea Schefer, Aaron Lee, Loren James Anderson, and Kristina Miller, "Low-Fuel Deep Reinforcement Learning Spacecraft Control through Lookahead Drifting," submitted to the *2027 IEEE Aerospace Conference*. DOI to be added.

**Organization:** Air Force Research Laboratory (AFRL)

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Repository Contents](#repository-contents)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Command-Line Options](#command-line-options)
- [Acknowledgements](#acknowledgements)

## Overview

Fuel is a limited resource for spacecraft, and refueling in orbit is costly and carries risk. Fuel-efficient control is therefore an important consideration throughout mission design, including proximity operations such as docking.

RL has recently emerged as a promising approach for autonomous spacecraft control. Prior work has applied it to docking, landing, and orbital control, largely in simulation.

A common way to encourage fuel efficiency is to include a fuel penalty in the reward function. However, this approach can be sensitive to tuning and may reduce training stability.
Drifter addresses this through drifting, which refers to periods of the maneuver in which the spacecraft applies no thrust.

Drifter-Refine and Drifter-Learn use a model of the dynamics to look ahead and determine how long the spacecraft can drift before additional thrust is needed. A model-free variant of Drifter-Learn, which does not require this model, is evaluated as an ablation study.

As a result, thrusting is limited to a small number of decision points. In addition, Drifter-Learn does not use a fuel penalty in its reward function. The paper presents two algorithms, along with ablation studies:

- **Drifter-Refine** is applied on top of a previously trained policy. It forces a drift whenever a simulated drift brings the deputy spacecraft closer to the chief spacecraft.
- **Drifter-Learn** is trained with forced drifting and a variable-length final time step in each episode, so that the Bellman backup covers a single time step per drift.
- Ablation studies cover model-free inference, action shielding, action noise, and sparse rewards.

The experiments use a Gymnasium implementation of a docking environment based on the [Aerospace SafeRL benchmark](https://github.com/act3-ace/SafeRL) (Ravaioli et al., IEEE Aerospace Conference, 2022). In this environment, a deputy spacecraft docks with a chief spacecraft, and the relative motion follows the Clohessy-Wiltshire-Hill (CWH) equations.

Each episode starts with the deputy 100 to 150 m from the chief. An episode ends when the deputy docks (within 0.5 m of the chief at a relative speed of at most 0.2 m/s), crashes, reaches an unsafe state, leaves the 200 m boundary, or reaches the maximum episode length. The thrust on each axis is limited to 1 N.

## Results

The table below shows the results reported in Table 3 of the paper, which compares the evaluated algorithms on the docking task.

The rows are the PPO baseline, Drifter-Refine, Drifter-Learn, the LQR controller, and four Drifter-Learn ablation studies: model-free testing, safety mechanisms, robustness to noise, and reward shaping.

Fuel is the total change in velocity (ΔV) used per episode in meters per second. Success is the percentage of evaluation episodes that end in docking. Time is the episode duration in seconds.

| Algorithm | Fuel (ΔV) | Success (%) | Time (s) |
|---|---|---|---|
| PPO | 19.01 [17.72, 20.10] | 99.50 [99.10, 99.90] | 109.43 [106.33, 113.23] |
| Drifter-Refine | 11.56 [10.55, 12.49] | 100.00 [100.00, 100.00] | 218.11 [215.44, 220.75] |
| Drifter-Learn | 2.34 [2.23, 2.49] | 95.90 [93.90, 97.50] | 799.76 [779.65, 820.33] |
| LQR | 1.39 [1.32, 1.45] | 100.00 [96.30, 100.00] | 1859.20 [1851.60, 1866.50] |
| Model-Free Testing | 2.82 [2.70, 2.94] | 90.40 [87.30, 92.90] | 661.10 [646.60, 679.90] |
| Safety Mechanisms | 2.26 [2.21, 2.31] | 96.70 [94.80, 98.30] | 808.77 [788.68, 828.97] |
| Robustness to Noise | 2.30 [2.26, 2.34] | 97.10 [96.00, 98.10] | 795.36 [775.34, 813.05] |
| Reward Shaping | 2.27 [2.23, 2.30] | 96.50 [95.20, 97.80] | 778.11 [766.38, 789.57] |

Each entry is the mean over 10 trained seeds on 100 held-out start states, with a 95% percentile-bootstrap confidence interval in brackets. LQR is a single deterministic controller, so its fuel and time intervals are bootstrapped over the 100 test episodes, and its success interval is a Wilson score interval.

## Repository Contents

| Purpose | Files |
|---|---|
| Docking environment for drifting | `drift_env.py` |
| Docking environment for direct docking (no drift) | `docking_env.py` |
| Drifter-Learn curriculum training | `drift_initial_trainer.py` |
| Evaluating a full curriculum chain of trained models | `drift_full_test.py`, `curriculum_evaluation.py` |
| Evaluating one checkpoint | `checkpoint_test.py` |
| Comparing two agents on paired episodes | `drift_vs_drift_test.py`, `drift_vs_nodrift_test.py` |
| Test-time ablations (safe action and action noise) | options in `drift_full_test.py` |
| Shared evaluation helpers | `evaluation_utilities.py` |
| Fixed evaluation start states | `make_test_set.py` and `test_sets/` |

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/DeepQZero/drifter.git
cd drifter
```

### Step 2: (Recommended) Create and Activate a Python Virtual Environment (verified with Python 3.14.5)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

Run this command from the repository folder, with the virtual environment active. It installs the exact package versions listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, install the direct dependencies yourself. These are the same packages and versions as in `requirements.txt`:

```bash
pip install gymnasium==1.3.0 stable-baselines3==2.9.0 torch==2.13.0 numpy==2.5.1 scipy==1.18.0 matplotlib==3.11.1 tensorboard==2.21.0 onnx==1.22.0 onnxruntime==1.28.0
```

| Package | Version | Used for |
|---|---|---|
| `gymnasium` | 1.3.0 | Environment interface for the docking environments |
| `stable-baselines3` | 2.9.0 | PPO training |
| `torch` | 2.13.0 | Neural networks used by PPO |
| `numpy` | 2.5.1 | Numerical computation |
| `scipy` | 1.18.0 | Numerical integration of the spacecraft dynamics |
| `matplotlib` | 3.11.1 | Trajectory and diagnostic plots |
| `tensorboard` | 2.21.0 | Training logs |
| `onnx`, `onnxruntime` | 1.22.0, 1.28.0 | Exporting and verifying trained policies in ONNX format (optional for training and evaluation) |

### Step 4: Confirm the Installation

```bash
python -c "import drift_env, docking_env, evaluation_utilities; print('Installation OK')"
```

If this prints `Installation OK`, the environments and evaluation helpers import correctly.

## Training

The steps below train Drifter-Learn and save the checkpoints that are used for evaluation.

### Drifter-Learn

#### Step 1: Choose the Training Settings

Open `drift_initial_trainer.py` and review the constants at the top of the file. The main settings are:

| Setting | Default | Description |
|---|---|---|
| `RUN_NAME` | `"drift_curriculum_PPO"` | Checkpoint folder name prefix |
| `NUM_STAGES` | 9 | Number of curriculum stages (0 to 8), from the closest start to the full range |
| `SEED` | `None` | Training seed; if `None`, a random seed is chosen and printed so the run can be reproduced |
| `BATCH_ID` | `None` | Optional tag that groups a run with others, recorded in `run_metadata.json` |
| `N_ENVS` | 4 | Number of parallel environments, roughly the number of CPU cores used |
| `TIMESTEPS_PER_EPOCH` | 25,000 | Training timesteps between each evaluation and checkpoint save |
| `MAX_EPOCHS_PER_STAGE` | 10 | Epochs after which a stage that has not reached its pass threshold is abandoned |
| `TEST_EPISODES` | 1,000 | Episodes used to measure the dock rate after each epoch |

The environment flags in this file are the ablation options. All of them are off by default, which is the baseline configuration.

#### Step 2: Run the Trainer

```bash
python drift_initial_trainer.py
```

The seed and batch tag can also be set from the command line:

```bash
python drift_initial_trainer.py --seed <seed> --batch-id <batch-id>
```

- Each stage trains in epochs and is evaluated after every epoch; training advances to the next stage once the stage reaches its dock-rate target
- Checkpoints are saved to `data/checkpoints/drift_curriculum_PPO_<run>/`, and the run number is assigned automatically as the lowest number not already in use
- The seed, flags, and per-stage configuration are recorded in `run_metadata.json` in the run folder

#### Checkpoint Naming

```
data/checkpoints/drift_curriculum_PPO_<run>/
    drift_curriculum_ppo_model_<run>_<stage>_<epoch>.zip
```

Here `<run>` is the run number, `<stage>` is the curriculum stage, and `<epoch>` is the epoch within that stage. A single stage is not a complete docking policy, so the evaluation scripts chain the stages of one run.

## Evaluation

### Test Sets

All paper results use the fixed start-state set `test_sets/standard_100_v2.json`. It contains 100 start states. In each one, the deputy starts 100 to 150 m from the chief with an initial speed between 0.10 and 0.50 m/s, measured as the magnitude of the relative velocity.

Each velocity component is sampled between -0.5 and 0.5 m/s. A start state is rejected if its speed exceeds the safe speed limit at that distance.

Each file in `test_sets/` records its seed and sampling bounds, along with a fingerprint that is checked whenever the file is loaded. The fingerprint of `standard_100_v2` is `737dce4df767`.

The seed and bounds of this set are the defaults of `make_test_set.py`. It can therefore be regenerated under a new name and checked against the fingerprint with:

```bash
python make_test_set.py --name <new-name> --expect-fingerprint 737dce4df767
```

### Drifter-Learn

#### Step 1: Choose the Checkpoints

Open `drift_full_test.py` and set `CHECKPOINT` near the top of the file. The supported forms are:

```python
# Newest checkpoints in the most recent run folder
CHECKPOINT = "latest"

# Checkpoints in a specific run folder
CHECKPOINT = "run:drift_curriculum_PPO_<run>"

# Glob pattern inside the checkpoint folder
CHECKPOINT = "pattern:drift_curriculum_ppo_model_<run>_*.zip"
```

To select exact checkpoint files, set `USE_MANUAL_CHECKPOINTS = True` and list the paths in `MANUAL_MODEL_PATHS`, from hardest to easiest stage.

#### Step 2: Choose the Evaluation Settings

| Setting | Default | Description |
|---|---|---|
| `NUM_EPISODES` | 30 | Number of episodes to run |
| `TEST_SET` | `test_sets/standard_100_v2.json` | Fixed start states, so every run faces identical episodes |
| `SEED` | `None` | Episode seed; if `None`, a random seed is chosen and printed |
| `USE_SAFE_ACTION` | `False` | Safe-action ablation: replace the model action with a one-step lookahead safety check |
| `USE_ACTION_NOISE` | `False` | Action-noise ablation: multiply every action by random noise in [0.95, 1.05] |
| `SHOW_PLOTS` | `False` | Open a window for each plot; if `False`, plots are saved to `saved_figures/` |

#### Step 3: Run the Evaluation

```bash
python drift_full_test.py
```

- The script starts with the hardest-stage model and switches to the next easier stage model each time the current one docks the spacecraft
- Only the final stage determines whether an episode is a win or a loss
- A per-episode results file is written to `saved_data/`, and trajectory plots are written to `saved_figures/`

### Other Evaluation Scripts

- `checkpoint_test.py` evaluates one checkpoint
- `curriculum_evaluation.py` evaluates a curriculum chain with a fixed seed
- `drift_vs_drift_test.py` and `drift_vs_nodrift_test.py` compare two agents on paired episodes

Each script is configured through constants at the top of the file.

Checkpoints, training data, and evaluation outputs are written to `data/`, `saved_data/`, `saved_results/`, and `saved_figures/`. These folders are not version controlled.

## Command-Line Options

`drift_full_test.py` takes no command-line flags. It is configured through the constants at the top of the file, as described above.

The following scripts accept flags:

| Script | Flag | Description |
|---|---|---|
| `drift_initial_trainer.py` | `--seed`, `-s` | Training seed, for reproducibility |
| | `--batch-id` | Tag that groups this run with others, recorded in `run_metadata.json` |
| `make_test_set.py` | `--num-states` | Number of start states to generate |
| | `--name` | Name of the set, which is also the file name in `test_sets/` |
| | `--seed` | Seed used to generate the start states |
| | `--max-vel-bound` | Limit on each velocity component of a start state, in meters per second, for the default distribution |
| | `--distribution` | Sampling distribution: `uniform_cube` (default), `space_controls` (sampling from Space Controls (2024)), or `saferl` (sampling from SafeRL (2022)) |
| | `--expect-fingerprint` | Stop without writing the file if the generated set has a different fingerprint |
| | `--overwrite` | Allow replacing an existing set with the same name; without it, the script refuses to overwrite |
| `checkpoint_test.py` | `--mode` | Agent mode: `drift`, `nodrift`, or `auto` (detect from the checkpoint file name); overrides `AGENT_MODE` in the file |
| | `--no-fix-unsafe` | Disable the patch that prevents premature unsafe terminations during evaluation |
| `rename_checkpoints.py` | `--apply` | Rename the files; without it, the script only reports which files it would rename |

The defaults for each script are constants at the top of its file. Example usage:

```bash
python make_test_set.py --num-states <number> --name <name> --seed <seed>
python checkpoint_test.py --mode <drift|nodrift|auto>
python rename_checkpoints.py --apply
```

## Acknowledgements

The first four authors completed this work as part of the 2025 Air Force Research Laboratory (AFRL) Scholars Program, funded by the Universities Space Research Association (USRA). Loren Anderson completed some of the experimentation for this work while employed by Huntington Ingalls Industries (HII); the current version of this work is not affiliated with HII in any form.

We thank Steven Tran, who interned in the 2025 AFRL Scholars Program, for helpful discussions.

The authors wish to acknowledge the AFRL Space Vehicles Directorate for their support of this work. *Space Superiority Modeling, Simulation, and Analyses (MS&A), and Applications Solutions (DTIC FA807525F0046).*

DISCLAIMER: The views expressed are those of the authors and do not reflect the official guidance or position of the United States Government, the Department of Defense or the United States Air Force.
