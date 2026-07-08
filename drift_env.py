"""
drift_env.py

Drift-assisted docking environment that combines CWH dynamics with DriftTrainEnv
and DriftTestEnv wrappers. At each step, the det_drift() method analytically
verifies whether drifting from the current state reaches the dock; if successful,
the episode is credited and terminates.

Utilized by drift_initial_trainer.py and the associated drift evaluation scripts.
"""

import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import norm as vec_norm


# ===================== SafeRL comparison options =====================
# Optional switches to mirror the AFRL/ACT3 SafeRL docking baseline (IEEE
# Aerospace 2022).
# Mirrors docking_env.py's block so the drift agent can be A/B compared
# against SafeRL. Usually defaults to False (current drifter behavior). Note
# the drift env already enforces the velocity constraint via is_unsafe, so
# it has no separate constraint flag.
# SafeRL references: saferl/aerospace/tasks/docking/processors.py and
# saferl/environment/tasks/processor/reward.py.

# True: append [speed, max_vel_limit] to the obs (SafeRL DockingObservation).
# False: 7-element obs [x, y, z, vx, vy, vz, timestep].
SAFERL_OBS = False

# True: exponential distance-change reward. False: linear approach reward.
SAFERL_EXP_DIST_REWARD = False
SAFERL_DIST_PIVOT = 100.0   # SafeRL 'pivot' (m): closing reward doubles every pivot
SAFERL_DIST_SCALE = 2.0     # SafeRL 'c'

# True: subtract a delta-V fuel penalty each step. False: no fuel term.
SAFERL_DELTA_V_PENALTY = False
SAFERL_DELTA_V_SCALE = 0.01  # SafeRL ProportionalRewardProcessor 'scale' on delta_v


class SpaceCraftDockingEnv3D(gym.Env):
    """
    The base docking environment in Gymnasium format.
    # TODO rename attributes sometime

    Attributes:
        fixed_start (bool): If the environment start should be fixed or not.
        fixed_state (numpy.ndarray): Fixed start state of the environment.
        reward_structure (str): "dense" if dense reward, "sparse" otherwise.
        max_episode_len (int): Maximum episode length.
        max_lookahead_len (int): Maximum number of lookahead time steps.
        max_boundary_box (int): Maximum distance between chief and deputy.
        max_control (float): Maximum thrust in Newtons allowed per thruster
        each time step.
        max_dv (float): Maximum fuel of deputy in dV. # TODO check
        pos_thresh (float): Maximum distance between chief and deputy for a
            successful dock.
        speed_thresh (float): Maximum relative speed between chief and deputy
            for a successful dock.
        min_init_pos_bound (float): Minimum relative starting distance.  # TODO
        max_init_pos_bound (float): Maximum relative starting distance.  # TODO
        step_len (int): Length of each step in seconds.  # TODO float?
        fuel_used (float): Fuel used so far, set to 0 in reset if None,
            used during resetting with fixed starts.
        time_step (int): Time step in seconds, set to 0 in reset if None,
            used during resetting with fixed starts.
        drift_step_len (int): Length of each drift step in seconds.  # TODO
    """
    def __init__(self,
                 fixed_start=False,
                 fixed_state=np.array([100, 0, 0, 0, 0, 0, 0]),
                 reward_structure="dense",
                 max_episode_len=10_000,
                 max_lookahead_len=1_000,
                 max_boundary_box=200.0,
                 max_control=1.0,
                 max_total_dv=1_000.0,
                 pos_thresh=0.5,
                 speed_thresh=0.2,
                 min_init_pos_bound=100.0,
                 max_init_pos_bound=150.0,
                 max_init_vel_bound=0.5,
                 step_len=1,
                 drift_step_len=1,
                 fuel_used=None,
                 time_step=None,
                 saferl_obs=SAFERL_OBS,
                 saferl_exp_dist_reward=SAFERL_EXP_DIST_REWARD,
                 saferl_delta_v_penalty=SAFERL_DELTA_V_PENALTY
                 ) -> None:
        self.fixed_start = fixed_start
        self.fixed_state = fixed_state
        self.reward_structure = reward_structure
        self.max_episode_len = max_episode_len
        self.lookahead_len = max_lookahead_len
        self.max_boundary_box = max_boundary_box
        self.u_max = max_control
        self.max_dv = max_total_dv
        self.dock_dist = pos_thresh
        self.dock_speed = speed_thresh
        self.min_start_dist = min_init_pos_bound
        self.max_start_dist = max_init_pos_bound
        self.max_start_speed = max_init_vel_bound
        self.step_len = step_len
        self.drift_step_len = drift_step_len
        self.fuel_used = fuel_used
        self.time_step = time_step

        self.n = 0.001027 # Mean motion of chief orbit (rad/s)
        self.m = 12  # Deputy mass (kg)
        self.time_penalty = -0.005  # TODO originally 0.0005
        self.dist_coeff = -0.005  # TODO originally 0.0005

        # SafeRL comparison options (see constants for details).
        self.saferl_obs = saferl_obs
        self.saferl_exp_dist_reward = saferl_exp_dist_reward
        self.saferl_delta_v_penalty = saferl_delta_v_penalty
        self.saferl_dist_scale = SAFERL_DIST_SCALE
        self.saferl_dist_a = np.log(2.0) / SAFERL_DIST_PIVOT  # closing reward doubles every pivot m
        self.saferl_delta_v_scale = SAFERL_DELTA_V_SCALE
        self.step_delta_v = 0.0  # Delta-V of the most recent step, set in step()

        self.action_space = spaces.Box(
            low=np.array([-self.u_max]*3),
            high=np.array([self.u_max]*3)
        )
        # 7 elements by default; 9 when saferl_obs appends [speed, max_vel_limit].
        obs_dim = 9 if self.saferl_obs else 7
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*obs_dim),
            high=np.array([np.inf]*obs_dim)
        )
        self.state = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Standard Gymnasium reset function returning start state and info."""
        super().reset(seed=seed)
        info = {}
        if self.fixed_start:
            self.state = np.copy(self.fixed_state)
        else:
            self.state = self.sample_state_space()
        self.fuel_used = 0
        self.time_step = 0
        self.step_delta_v = 0.0
        return self._get_obs(), info

    def velocity_limit(self, distance: float) -> float:
        """Distance-scaled safe speed limit (SafeRL DockingVelocityLimit:
        0.2 + slope * n * dist). Shared by is_unsafe, the start-state
        rejection, and the optional SafeRL observation.
        """
        return 0.2 + 2 * self.n * distance

    def _get_obs(self) -> np.ndarray:
        """Return the observation: the 7-element state by default, or with
        [speed, max_vel_limit] appended when saferl_obs is on.
        """
        if not self.saferl_obs:
            return self.state
        distance = vec_norm(self.state[0:3])
        speed = vec_norm(self.state[3:6])
        max_vel_limit = self.velocity_limit(distance)
        return np.concatenate([self.state, np.array([speed, max_vel_limit])])

    def sample_state_space(self) -> np.ndarray:
        """Samples and returns start state."""
        low = np.array([-self.max_start_dist] * 3 + [-self.max_start_speed] * 3)
        high = np.array([self.max_start_dist] * 3 + [self.max_start_speed] * 3)
        # Sample directly from the env's own seeded generator instead of
        # Box.sample(), since Box has no public way to reuse an external
        # generator in this Gymnasium version.
        sampled_state = self.np_random.uniform(low=low, high=high)
        rel_dist = vec_norm(sampled_state[0:3])
        rel_speed = vec_norm(sampled_state[3:6])
        if ((rel_dist < self.min_start_dist) or
            (rel_dist > self.max_start_dist) or
            (rel_speed > self.velocity_limit(rel_dist))):
            return self.sample_state_space()  # TODO recursion error maybe?
        return np.concatenate([sampled_state, np.array([0])])

    def step(self, action: np.ndarray, drift=False) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Standard Gymnasium step function."""
        self.time_step += 1  # TODO remove?
        old_state = np.copy(self.state)  # deep copy
        self.state = self.propagate(action, drift)
        self.fuel_used += vec_norm(action)/self.m * self.step_len  # TODO
        # TODO only allow fuel used to be a certain amount? Check paper.
        # SafeRL-style per-step delta-V (L1 thrust / mass * step).
        self.step_delta_v = float(np.sum(np.abs(action))) / self.m * self.step_len
        reward, terminated, truncated = self.rewards(old_state)
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def propagate(self, action: np.ndarray, drift: bool=False) -> np.ndarray:
        """Computes new state."""
        dt = self.step_len if not drift else self.drift_step_len
        t_span = (self.time_step, self.time_step + dt)
        result = solve_ivp(
            self.dynamics,
            t_span,
            self.state[0:6],
            args=(action,),
            method='RK45',
            rtol=1e-5,
            atol=1e-8
        )
        if not result.success:
            return self.state
        new_state = result.y[:, -1]
        return np.concatenate([new_state, np.array([1 + self.state[6]])])

    def dynamics(self, t, x, u):  # TODO type signature
        """Computes new state using CWH equations."""
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [(3 * self.n ** 2), 0, 0, 0, 2 * self.n, 0],
                      [0, 0, 0, -2 * self.n, 0, 0],
                      [0, 0, - (self.n ** 2), 0, 0, 0]])
        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1 / self.m, 0, 0],
                      [0, 1 / self.m, 0],
                      [0, 0, 1 / self.m]])
        return A.dot(x) + B.dot(u)

    def rewards(self, last_state: np.ndarray) -> tuple[float, bool, bool]:
        """Computes step reward, termination, and truncation conditions."""
        tot_step_rew = 0

        docked = self.is_docked()
        crashed = self.is_crashed()
        out_of_time = self.is_out_of_time()
        out_of_fuel = self.is_out_of_fuel()
        out_of_bounds = self.is_out_of_bounds()
        unsafe = self.is_unsafe()

        term = (docked or crashed or out_of_bounds or out_of_fuel or unsafe or
                out_of_time)
        trunc = False

        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0
        if self.reward_structure == "dense":  # TODO det when training refiner
            if docked:
                tot_step_rew += 1
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time or unsafe:
                tot_step_rew += -1
            current_distance = float(vec_norm(self.state[0:3]))  # TODO float?
            prev_distance = float(vec_norm(last_state[0:3]))
            if self.saferl_exp_dist_reward:
                # SafeRL exponential distance-change reward (stronger near target).
                prox_penalty = self.saferl_dist_scale * (
                    np.exp(-self.saferl_dist_a * current_distance)
                    - np.exp(-self.saferl_dist_a * prev_distance)
                )
            else:
                prox_penalty = (self.dist_coeff *
                                (current_distance - prev_distance))
            tot_step_rew += (prox_penalty + self.time_penalty)
            if self.saferl_delta_v_penalty:
                # SafeRL delta-V fuel penalty (scale * step delta-V).
                tot_step_rew += -self.saferl_delta_v_scale * self.step_delta_v
        return tot_step_rew, term, trunc

    def is_docked(self) -> bool:
        """Determines if deputy is docked."""
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) < self.dock_speed)

    def is_crashed(self) -> bool:
        """Determines if deputy has crashed."""
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) >= self.dock_speed)

    def is_out_of_fuel(self):
        """Determines if deputy is out of fuel."""
        return self.fuel_used > self.max_dv

    def is_out_of_bounds(self):
        """Determines if deputy is out of bounds."""
        return vec_norm(self.state[0:3]) > self.max_boundary_box

    def is_out_of_time(self):
        """Determines if episode is out of time."""
        return self.state[6] >= self.max_episode_len  # TODO > or >=?

    def is_unsafe(self):
        current_distance = float(vec_norm(self.state[0:3]))  # TODO float?
        current_speed = float(vec_norm(self.state[3:6]))
        speed_limit = self.velocity_limit(current_distance)
        return current_speed - speed_limit > 0

    def only_oot(self):
        return (not (self.is_docked() or self.is_crashed() or
                     self.is_out_of_bounds() or self.is_out_of_fuel()
                     or self.is_unsafe())) and self.is_out_of_time()

    def term_other_than_oot(self):
        return (self.is_docked() or self.is_crashed() or
                self.is_out_of_bounds() or self.is_out_of_fuel()
                or self.is_unsafe())

    def close(self) -> None:
        """Standard Gymnasium close function."""
        pass


class DriftTrainEnv(gym.Env):
    """Wrapper for docking environment that looks ahead each time 
    step to determine if docking condition can be achieved by drifting.
    """
    def __init__(self, **kwargs) -> None:
        self.env = SpaceCraftDockingEnv3D(**kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Standard Gymnasium reset function returning start state and info."""
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Standard Gymnasium step function."""
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        if (not done) or (self.env.only_oot()):
            is_drift, the_time = self.det_drift()
            if is_drift:
                rew += 10
                term = True
        return obs, rew, term, trunc, info

    def det_drift(self) -> tuple[bool, int]:
        """Determines if drifting leads to a correct docking."""
        new_env = copy.deepcopy(self.env)
        t = 0
        for j in range(self.env.lookahead_len):
            drift_action = np.array([0.0, 0.0, 0.0])
            t += 1
            new_env.step(drift_action, True)
            if new_env.term_other_than_oot():
                return new_env.is_docked(), t
        return False, t


class DriftTestEnv(gym.Env):
    """
    Wrapper for docking environment that looks ahead each time step to
    determine if docking condition can be achieved by drifting.
    """
    def __init__(self, **kwargs) -> None:
        self.env = SpaceCraftDockingEnv3D(**kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.is_drifting = False

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Standard Gymnasium reset function returning start state and info."""
        self.is_drifting = False
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Standard Gymnasium step function."""
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        if self.env.is_docked():
            print('WIN!')
        if (not done) or (self.env.only_oot()):
            if not self.is_drifting:
                is_drift, the_time = self.det_drift()
                if is_drift:
                    self.is_drifting = True
        return obs, rew, term, trunc, info

    def det_drift(self) -> tuple[bool, int]:
        """Determines if drifting leads to a correct docking."""
        new_env = copy.deepcopy(self.env)
        t = 0
        for j in range(self.env.lookahead_len):
            drift_action = np.array([0.0, 0.0, 0.0])
            t += 1
            new_env.step(drift_action, True)
            if new_env.term_other_than_oot():
                return new_env.is_docked(), t
        return False, t
