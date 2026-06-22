# Drifter

A Proximal Policy Optimization (PPO)-based reinforcement learning (RL) system for autonomous spacecraft rendezvous and docking using Clohessy-Wiltshire-Hill (CWH) orbital dynamics. The deputy spacecraft learns to approach and dock with a chief spacecraft. We implement two primary training approaches: drift-assisted docking (Drifter-Learn) and direct docking without drift.

## Authors

**Organization:** Air Force Research Laboratory (AFRL)

**Authors:**
- Loren James Anderson
- Marcos Sanson
- Nikita Agrawal
- Aaron Lee
- Andrea Schefer

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Ideas](#ideas)
- [TODO](#todo)
- [Definitions](#definitions)

---

## Repository Structure

```
drifter/
├── .gitignore
├── README.md
│
├── drift_env.py                  # Drift-assisted docking environment (DriftTrainEnv, DriftTestEnv)
├── docking_env.py                # Direct docking environment, no drift mechanic
│
├── initial_trainer.py            # Curriculum training for the drift agent
├── initial_trainer2.py           # Alternate curriculum trainer with additional stats
├── nodrift_initial_trainer.py    # Curriculum training for the direct docking agent
│
├── train.py                      # Single-run PPO training with checkpoint callbacks
├── nodrift_train.py              # Single-run PPO training, no drift
│
├── test.py                       # Quick single-model evaluation script
├── part_test.py                  # Partial evaluation script
├── question.py                   # Scratch/experimental script
│
├── full_test.py                  # Manual end-to-end curriculum chain evaluation
├── checkpoint_test.py            # Single checkpoint evaluation with trajectory plots
├── curriculum_evaluation.py      # Automated curriculum chain evaluation
├── evaluation_utilities.py       # Shared checkpoint resolution and evaluation helpers
│
└── data/
    └── checkpoints/
        ├── safe_PPO_1/           # Drift curriculum run 1
        │   ├── safe_ppo_model_0_1_0.zip
        │   ├── safe_ppo_model_1_1_0.zip
        │   ├── ...
        │   └── safe_ppo_model_9_1_0.zip
        ├── safe_PPO_8/           # Example: drift curriculum run 8
        │   ├── safe_ppo_model_1_8_0.zip
        │   ├── safe_ppo_model_1_8_1.zip   # epoch 1 (if threshold not met at epoch 0)
        │   └── ...
        ├── nodrift_curriculum_PPO_1/
        │   ├── nodrift_ppo_model_0_1_0.zip
        │   └── ...
        └── drifter_PPO_1/        # Single-run (non-curriculum) checkpoints
            ├── drifter_ppo_25000_steps.zip
            ├── drifter_ppo_50000_steps.zip
            └── ...
```

Checkpoint files and training data are excluded from version control via `.gitignore`.

---

## Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/DeepQZero/drifter.git
cd drifter
```

### Step 2: (Recommended) Create and Activate a Python Virtual Environment (Python 3.9+)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install stable-baselines3 gymnasium numpy scipy matplotlib
```

---

## Training

- Run `initial_trainer.py`; model id can be changed at the bottom of the script
  - Checkpoints save automatically to `data/checkpoints/safe_PPO_<run>/`
  - Training saves a checkpoint every 25,000 timesteps by default; can change the timesteps to adjust how often checkpoints are saved and tested

- For direct docking (no drift), run `nodrift_initial_trainer.py` instead
  - Can change `RUN_NAME` and `SAVE_ALL_EPOCHS` at the top of the file
    - `SAVE_ALL_EPOCHS = False` keeps only the final passing checkpoint per stage

### nodrift_initial_trainer.py Configuration

| Variable | Default | Description |
|---|---|---|
| `RUN_NAME` | `"nodrift_curriculum_PPO"` | Checkpoint folder name prefix |
| `NUM_STAGES` | `6` | Number of curriculum stages to run |
| `TIMESTEPS_PER_EPOCH` | `25_000` | Training steps per save-and-test cycle |
| `TEST_EPISODES` | `1_000` | Episodes used to measure dock rate after each save |
| `SAVE_ALL_EPOCHS` | `False` | Keep every epoch checkpoint, or only the passing one |

---

## Checkpoint Naming

**Drift curriculum:**
```
data/checkpoints/safe_PPO_{run}/safe_ppo_model_{stage}_{run}_{epoch}.zip
```
Example: `safe_PPO_8/safe_ppo_model_1_8_0.zip` is run 8, stage 1, epoch 0.

**Direct docking curriculum:**
```
data/checkpoints/nodrift_curriculum_PPO_{run}/nodrift_ppo_model_{stage}_{run}_{epoch}.zip
```

**Single-run (train.py):**
```
data/checkpoints/drifter_PPO_{run}/drifter_ppo_{timesteps}_steps.zip
```

The run number in each folder name increments automatically so old runs are never overwritten.

---

## Evaluation

- Individual models can be tested in `test.py`; make sure to change the curriculum number
  - Update the model path and curriculum index at the top of the script
  - Counts wins by checking if the final reward exceeds 9, which indicates a drift success
  - Prints starting distance, win count, mean timesteps, and mean fuel per episode

- Individual models can also be tested in `checkpoint_test.py` for more detailed output
  - Set `CHECKPOINT` at the top of the file to point to your model
  - Outputs win rate, failure breakdown, a 3-dimensional (3D) trajectory plot, and a JavaScript Object Notation (JSON) results file

- Combined models can be tested in `full_test.py`; place your models at the top
  - Chains models from hardest to easiest stage within a single episode
  - Only the final stage result counts as a win or loss
  - `curriculum_evaluation.py` is a newer automated version of the same idea with configurable checkpoint resolution

### Checkpoint Formats for Testing

```python
CHECKPOINT = "latest"                         # newest checkpoint in the most recent run folder
CHECKPOINT = "run:safe_PPO_8"                 # newest checkpoint inside a specific run folder
CHECKPOINT = "pattern:safe_ppo_model_*_8.zip" # glob pattern inside the checkpoint root
CHECKPOINT = r"data\checkpoints\safe_PPO_8\safe_ppo_model_9_8_0.zip"  # exact path
```

### Evaluation Output Files

After running `checkpoint_test.py` or `curriculum_evaluation.py`, results are saved automatically:

- `saved_figures/` - 3D trajectory plots as Portable Network Graphics (PNG) files, one per evaluation run
- `saved_results/` - JSON summary files containing win rate, fuel statistics, and per-episode results

---

## Troubleshooting

- Action isn't set to deterministic for testing
  - Pass `deterministic=True` to `model.predict()` during evaluation; without it the agent samples randomly instead of using its best learned action

- Agent is docking and not drifting to dock (rewards are smaller for docking than drifting)
  - The drift reward is +10 and the direct dock reward is +1, so a well-trained agent should prefer to drift when a valid drift trajectory is available
  - If the agent never drifts, `max_lookahead_len` may be too short to detect a valid drift path from the current position; try increasing it

- Gymnasium emits float32 warnings about Box observation bounds
  - These are harmless and can be ignored; they are caused by float64 bounds in the observation space definition and do not affect training or evaluation results

---

## Ideas

- Reduce entropy coefficient
  - A lower entropy coefficient later in training would likely encourage more deterministic, committed behavior; this could probably be implemented as a scheduled parameter that decreases over the course of training

- Make Partially Observable Markov Decision Process (POMDP) and remove time step
  - Removing the timestep counter from the state vector would make the problem partially observable, which is probably more realistic for onboard deployment; not sure whether this would help or hurt sample efficiency

- Curricula could be more focused at start states (e.g. near 2.5, 10, 50, 150)
  - Sampling start positions near the stage transition distances rather than uniformly across the full range may help the agent generalize across stages more reliably; this would likely require some per-stage tuning

- Export to Open Neural Network Exchange (ONNX) or TorchScript for deployment
  - Frozen inference models could probably be exported for use on onboard hardware; this has not been tested and may require changes to the environment wrapper

---

## TODO

- Do `env.unwrapped`
  - Replace direct `env.env` attribute access with `env.unwrapped` to follow the Gymnasium application programming interface (API) properly

---

## Definitions

**API (Application Programming Interface):**
A set of rules and conventions that software components use to communicate with each other. In this project, the Gymnasium API defines how environments expose observations, actions, and rewards to training scripts.

**Chief/Deputy:**
Terms used to describe the primary (chief) and secondary (deputy) spacecraft in relative motion. The deputy maneuvers relative to the stationary chief.

**CWH (Clohessy-Wiltshire-Hill) Equations:**
A set of linearized differential equations describing the relative motion of one spacecraft (the deputy) with respect to another (the chief) in a circular orbit. Used throughout this project to propagate the deputy's position and velocity each timestep.

**Curriculum Learning:**
A training method where an agent learns tasks in order from easy to hard. In this project, the agent first trains at close range with a large docking target, then progressively trains at longer ranges with tighter docking requirements.

**Delta-V (dV):**
A measure of the total change in velocity a spacecraft uses for maneuvering. Used here as a proxy for fuel consumption.

**Drifting:**
A period where the spacecraft applies zero thrust and coasts along its natural orbital trajectory. The key point of this project is that drifting can sometimes complete a docking scenario with less fuel than actively thrusting.

**JSON (JavaScript Object Notation):**
A lightweight text format for storing and exchanging data. Used in this project to save evaluation results such as win rates, fuel statistics, and per-episode outcomes.

**Lookahead:**
A model-based check performed each timestep to determine whether drifting from the current state would lead to a successful dock within a fixed number of steps. If the lookahead returns true, the agent stops thrusting and drifts.

**ONNX (Open Neural Network Exchange):**
An open format for representing machine learning models, designed to make it easier to move models between different software frameworks and hardware platforms.

**POMDP (Partially Observable Markov Decision Process):**
A framework for sequential decision-making where the agent cannot directly observe the full environment state. Removing the timestep counter from the state vector would make this problem a POMDP.

**PPO (Proximal Policy Optimization):**
A popular policy gradient algorithm used to train the agents in this project. PPO updates the agent's policy using small, controlled steps to keep training stable. It is widely used in robotics and simulated environments because it is reliable and relatively simple to tune.

**RL (Reinforcement Learning):**
A machine learning approach where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties. Currently, all agents in this project are trained using reinforcement learning.

**RPO (Rendezvous and Proximity Operations):**
A set of spaceflight maneuvers where one spacecraft approaches and potentially docks with another object in orbit. This project focuses on the final docking phase of rendezvous and proximity operations (RPO).

**State Vector:**
The numerical representation of the environment passed to the agent at each timestep. In this project the state is a 7-element vector: relative position (x, y, z), relative velocity (vx, vy, vz), and a timestep counter.
