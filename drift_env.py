"""Spacecraft docking environment where agents learn to use coast phases.

Uses Clohessy-Wiltshire-Hill orbital dynamics. At each step, checks if
zero-thrust coasting would reach the dock; if so, the episode ends
successfully. This allows agents to learn fuel-efficient docking by
cutting thrust and coasting when possible.

Used by drift_initial_trainer.py and the drift evaluation scripts.
"""

import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import norm as vec_norm

from evaluation_utilities import propagation_matrices

# ===================== Comparison options =====================
# Optional switches to mirror the AFRL/ACT3 SafeRL docking baseline (IEEE
# Aerospace 2022) for A/B comparison. Usually defaults to False (current
# drifter behavior); flip here or via the matching constructor argument.
# SafeRL references: saferl/aerospace/tasks/docking/processors.py and
# saferl/environment/tasks/processor/reward.py.
# Also includes Space Controls (2024, arXiv:2405.12355) comparison flags.

# All comparison flags default to False. Recorded in run_metadata.json and
# auto-detected by auto_label_checkpoints().
# Prefix says the source: SAFERL_ from SafeRL (2022), SPACE_CONTROLS_
# from the 2024 paper, CUSTOM_ is this project's.

# True: append [speed, max_vel_limit] to the observation.
# False: 7-element observation [x, y, z, vx, vy, vz, timestep].
SAFERL_OBS = False

# True: exponential distance-change reward.
# False: linear approach reward.
SAFERL_EXP_DIST_REWARD = False
SAFERL_DIST_PIVOT = 100.0   # closing reward doubles every this many meters
SAFERL_DIST_SCALE = 2.0

# True: subtract a Delta-v fuel penalty each step.
# False: no fuel term.
SAFERL_DELTA_V_PENALTY = False
SAFERL_DELTA_V_SCALE = 0.01

# True: on a win, add a bonus of up to +1 for finishing early.
# False: flat win reward only. Off by default.
# Applies to both a direct dock and a coast win.
SAFERL_SUCCESS_TIME_BONUS = False

# True: flat +2 win reward (SafeRL 2022 Table 8). False: our +1.
SAFERL_SUCCESS_REWARD = False
SAFERL_SUCCESS_REWARD_VALUE = 2.0

# True: budgeted velocity constraint. Violations accumulate a graded
# penalty; the episode fails once the total reaches SAFERL_VEL_BUDGET.
# False: the first violation ends the episode with a flat -1.
SAFERL_VEL_CONSTRAINT = False
SAFERL_VEL_VIOLATION_SCALE = -0.01
SAFERL_VEL_VIOLATION_BIAS = -0.01   # flat cost per violating step
SAFERL_VEL_BUDGET = -5.0            # episode fails when the penalty sum reaches this

# True: drop the timestep, matching DRL for Space Controls (2024)'s
# 6-element state [x, y, z, vx, vy, vz].
# False: keep the 7-element state.
SPACE_CONTROLS_OBS = False

# True: flat -0.01 per-step time penalty, matching DRL for Space Controls (2024).
# False: this environment's default of -0.005.
SPACE_CONTROLS_TIME_PENALTY = False
SPACE_CONTROLS_TIME_PENALTY_VALUE = -0.01

# True: flat -0.0005 per-step time penalty (this project's paper value).
# False: default -0.005.
PAPER_TIME_PENALTY = False
PAPER_TIME_PENALTY_VALUE = -0.0005

# True: no time penalty (SafeRL 2022 Table 8 has no time term).
# False: default -0.005, unless other overrides apply.
SAFERL_NO_TIME_PENALTY = False

# True: unsafe reward is 0 (this project's paper). False: default -1.
# Only applies when saferl_vel_constraint is off.
PAPER_UNSAFE_REWARD = False

# True: limit a start state's speed at 80% of the speed limit, matching
# DRL for Space Controls (2024)'s sampling.
# False: allow up to 100%, so some episodes start right at the limit.
SPACE_CONTROLS_INIT_VEL_FRACTION = False
SPACE_CONTROLS_INIT_VEL_SCALE = 0.8

# True: speed limit measured from the docking radius,
# 0.2 + 2n(dist - dock_dist), matching DRL for Space Controls (2024) Eq. 5.
# False: 0.2 + 2n*dist, matching SafeRL (2022) Eq. 17.
SPACE_CONTROLS_SPEED_LIMIT_OFFSET = False

# True: 2024's velocity penalty (scaled only, no flat bias); violations
# never end the episode. Takes priority over SAFERL_VEL_CONSTRAINT if both are on.
# False: use SAFERL_VEL_CONSTRAINT's form instead, or no constraint if that's off too.
SPACE_CONTROLS_VEL_CONSTRAINT = False

# The four flags below REPLACE their constructor argument instead of
# using it. Every curriculum stage supplies these keys directly.

# True: dock at DRL for Space Controls (2024)'s 10 m radius.
# False: use pos_thresh, normally 0.5 m (SafeRL 2022 Table 4).
SPACE_CONTROLS_DOCK_RADIUS = False
SPACE_CONTROLS_DOCK_RADIUS_VALUE = 10.0

# True: out-of-bounds at DRL for Space Controls (2024)'s 800 m.
# False: use max_boundary_box, normally 200 m.
SPACE_CONTROLS_MAX_DISTANCE = False
SPACE_CONTROLS_MAX_DISTANCE_VALUE = 800.0

# True: 40,000 m max distance (SafeRL 2022 Table 10, catches runaway only).
# False: 200 m (default). Overrides SPACE_CONTROLS_MAX_DISTANCE if both on.
SAFERL_MAX_DISTANCE = False
SAFERL_MAX_DISTANCE_VALUE = 40_000.0

# True: 0.1 N per axis, the low-thrust variant in DRL for Space Controls
# (2024)'s sweep.
# False: use max_control, normally 1.0 N.
SPACE_CONTROLS_LOW_THRUST = False
SPACE_CONTROLS_LOW_THRUST_VALUE = 0.1

# True: append a braking margin (distance minus stopping distance) to
# the observation, a "brake now" signal.
# False: observation unchanged. Not part of SafeRL; our addition.
CUSTOM_BRAKING_MARGIN_OBS = False

# True: graded proximity-scaled speed limit penalty (custom addition).
# False: no extra speed penalty. Pair with SAFERL_VEL_CONSTRAINT.
CUSTOM_VEL_PENALTY = False
CUSTOM_VEL_PENALTY_COEFF = -0.02  # matches docking_env.py's tuned value

# True: normalize obs (pos/100, vel/0.5, SafeRL 2022 Table 6).
# False: raw values (default, unlike docking_env.py). Keep True for SafeRL replicate.
NORMALIZE_OBS = False
OBS_POS_NORM = 100.0
OBS_VEL_NORM = 0.5

# Retry limit for start-state rejection sampling. Hitting this limit means
# the bounds are too tight.
MAX_START_SAMPLE_TRIES = 10000

# True: print "WIN!" on a successful dock.
# False: stay quiet (eval scripts print their own win/loss lines).
PRINT_WIN_MESSAGE = False

# True: exact closed-form CWH solution (faster, det_drift uses this).
# False: solve_ivp (RK45). Both agree to about 1e-11; speed setting only.
FAST_ANALYTIC_PROPAGATION = False


class DriftDockingEnv3D(gym.Env):
    """The base CWH docking environment (drift variant) in Gymnasium format.

    The drift agent's base environment, wrapped by DriftTrainEnv and
    DriftTestEnv. Distinct from docking_env.py's SpaceCraftDockingEnv3D,
    which the nodrift agents use.

    Attributes:
        fixed_start (bool): If the environment start should be fixed or not.
        fixed_state (numpy.ndarray): Fixed start state of the environment.
        reward_structure (str): "dense" if dense reward, "sparse" otherwise.
        max_episode_len (int): Maximum episode length. Also ends the
        episode via is_out_of_time().
        obs_time_norm (int): Divisor for the observation's timestep
        element only. Defaults to max_episode_len; set per stage during
        chain evaluation so obs[6] stays in the range the policy trained
        on. Does not affect termination.
        max_lookahead_len (int): Maximum number of lookahead time steps.
        max_boundary_box (int): Maximum distance between chief and deputy.
        max_control (float): Maximum thrust in Newtons allowed per thruster
        each time step.
        max_dv (float): Maximum propellant budget in m/s before the
            episode ends out-of-fuel, compared against fuel_used_l1
            (see is_out_of_fuel).
        pos_thresh (float): Maximum distance between chief and deputy for a
            successful dock.
        speed_thresh (float): Maximum relative speed between chief and deputy
            for a successful dock.
        min_init_pos_bound (float): Minimum relative starting distance from
            the chief, in meters.
        max_init_pos_bound (float): Maximum relative starting distance from
            the chief, in meters.
        step_len (int): Duration of each environment step, in seconds.
        fuel_used (float): Cumulative L2 Delta-v so far, what a single
            gimbaled thruster would have spent to fly the same
            trajectory. Reported as a secondary metric. Set to 0 in
            reset if None, used during resetting with fixed starts.
        fuel_used_l1 (float): Cumulative L1 Delta-v, proportional to
            propellant used by the three fixed per-axis
            thrusters. The primary fuel metric, and what the Delta-v
            reward penalty and the max_dv budget both use.
        time_step (int): Time step in seconds, set to 0 in reset if None,
            used during resetting with fixed starts.
        drift_step_len (int): Duration of each look-ahead drift (coast)
            step, in seconds.
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
                 saferl_delta_v_penalty=SAFERL_DELTA_V_PENALTY,
                 saferl_success_time_bonus=SAFERL_SUCCESS_TIME_BONUS,
                 saferl_success_reward=SAFERL_SUCCESS_REWARD,
                 custom_braking_margin_obs=CUSTOM_BRAKING_MARGIN_OBS,
                 custom_vel_penalty=CUSTOM_VEL_PENALTY,
                 saferl_vel_constraint=SAFERL_VEL_CONSTRAINT,
                 saferl_max_distance=SAFERL_MAX_DISTANCE,
                 saferl_no_time_penalty=SAFERL_NO_TIME_PENALTY,
                 normalize_obs=NORMALIZE_OBS,
                 space_controls_obs=SPACE_CONTROLS_OBS,
                 space_controls_time_penalty=SPACE_CONTROLS_TIME_PENALTY,
                 space_controls_init_vel_fraction=SPACE_CONTROLS_INIT_VEL_FRACTION,
                 space_controls_speed_limit_offset=SPACE_CONTROLS_SPEED_LIMIT_OFFSET,
                 space_controls_vel_constraint=SPACE_CONTROLS_VEL_CONSTRAINT,
                 space_controls_dock_radius=SPACE_CONTROLS_DOCK_RADIUS,
                 space_controls_max_distance=SPACE_CONTROLS_MAX_DISTANCE,
                 space_controls_low_thrust=SPACE_CONTROLS_LOW_THRUST,
                 paper_time_penalty=PAPER_TIME_PENALTY,
                 paper_unsafe_reward=PAPER_UNSAFE_REWARD,
                 ) -> None:
        self.fixed_start = fixed_start
        self.fixed_state = fixed_state
        self.reward_structure = reward_structure
        self.max_episode_len = max_episode_len
        # Timestep observation divisor; defaults to max_episode_len and may vary by stage.
        self.obs_time_norm = max_episode_len
        self.lookahead_len = max_lookahead_len
        # These four flags REPLACE their constructor argument.
        # saferl_max_distance > space_controls_max_distance.
        self.max_boundary_box = (
            SAFERL_MAX_DISTANCE_VALUE if saferl_max_distance
            else SPACE_CONTROLS_MAX_DISTANCE_VALUE if space_controls_max_distance
            else max_boundary_box)
        self.u_max = (
            SPACE_CONTROLS_LOW_THRUST_VALUE if space_controls_low_thrust
            else max_control)
        self.max_dv = max_total_dv
        self.dock_dist = (
            SPACE_CONTROLS_DOCK_RADIUS_VALUE if space_controls_dock_radius
            else pos_thresh)
        self.dock_speed = speed_thresh
        self.min_start_dist = min_init_pos_bound
        self.max_start_dist = max_init_pos_bound
        self.max_start_speed = max_init_vel_bound
        self.step_len = step_len
        self.drift_step_len = drift_step_len
        self.fuel_used = fuel_used
        self.fuel_used_l1 = 0.0
        self.time_step = time_step

        self.n = 0.001027 # Mean motion of chief orbit (rad/s)
        self.m = 12  # Deputy mass (kg)
        # Overrides: space_controls > paper > saferl_no_time_penalty; else default -0.005.
        if space_controls_time_penalty:
            self.time_penalty = SPACE_CONTROLS_TIME_PENALTY_VALUE
        elif paper_time_penalty:
            self.time_penalty = PAPER_TIME_PENALTY_VALUE
        elif saferl_no_time_penalty:
            self.time_penalty = 0.0
        else:
            self.time_penalty = -0.005
        self.paper_unsafe_reward = paper_unsafe_reward
        # Matches paper and docking_env.py's dist_coeff for fuel comparison.
        self.dist_coeff = -0.0005

        # SafeRL comparison options (see constants for details).
        self.saferl_obs = saferl_obs
        self.saferl_exp_dist_reward = saferl_exp_dist_reward
        self.saferl_delta_v_penalty = saferl_delta_v_penalty
        self.saferl_success_time_bonus = saferl_success_time_bonus
        self.saferl_success_reward = saferl_success_reward
        self.custom_braking_margin_obs = custom_braking_margin_obs
        self.custom_vel_penalty = custom_vel_penalty
        self.saferl_vel_constraint = saferl_vel_constraint
        self.normalize_obs = normalize_obs
        # Space Controls comparison options (see constants for details).
        self.space_controls_obs = space_controls_obs
        self.space_controls_init_vel_scale = SPACE_CONTROLS_INIT_VEL_SCALE if space_controls_init_vel_fraction else 1.0
        self.saferl_dist_scale = SAFERL_DIST_SCALE
        self.saferl_dist_a = np.log(2.0) / SAFERL_DIST_PIVOT  # closing reward doubles every pivot m
        self.saferl_delta_v_scale = SAFERL_DELTA_V_SCALE
        self.vel_penalty_coeff = CUSTOM_VEL_PENALTY_COEFF
        self.saferl_vel_violation_scale = SAFERL_VEL_VIOLATION_SCALE
        self.space_controls_speed_limit_offset = space_controls_speed_limit_offset
        self.space_controls_vel_constraint = space_controls_vel_constraint
        # space_controls_vel_constraint drops the flat per-violating-step
        # term; saferl_vel_constraint keeps it. Space Controls takes
        # priority when both are on.
        self.saferl_vel_violation_bias = (
            0.0 if space_controls_vel_constraint else SAFERL_VEL_VIOLATION_BIAS)
        self.saferl_vel_budget = SAFERL_VEL_BUDGET
        self.step_delta_v = 0.0  # Delta-v of the most recent step, set in step()
        self.vel_violation_reward_sum = 0.0  # Constraint budget used this episode

        self.action_space = spaces.Box(
            low=np.array([-self.u_max]*3, dtype=np.float32),
            high=np.array([self.u_max]*3, dtype=np.float32)
        )
        # 7 elements by default (6 with space_controls_obs), +2 for saferl_obs,
        # +1 for custom_braking_margin_obs. Matches docking_env.py's layout.
        obs_dim = 6 if self.space_controls_obs else 7
        if self.saferl_obs:
            obs_dim += 2
        if self.custom_braking_margin_obs:
            obs_dim += 1
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*obs_dim, dtype=np.float32),
            high=np.array([np.inf]*obs_dim, dtype=np.float32)
        )
        self.state = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to a starting state.

        Args:
            seed: Random seed for reproducibility.
            options: Unused, required by the Gymnasium interface.

        Returns:
            A tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        info = {}
        if self.fixed_start:
            self.state = np.copy(self.fixed_state)
        else:
            self.state = self.sample_state_space()
        self.fuel_used = 0
        self.fuel_used_l1 = 0.0
        self.time_step = 0
        self.step_delta_v = 0.0
        self.vel_violation_reward_sum = 0.0
        return self._get_obs(), info

    def velocity_limit(self, distance: float) -> float:
        """Distance-scaled safe speed limit.

        Shared by is_unsafe, the start-state rejection, and the optional
        SafeRL observation. Default is SafeRL (2022) Eq. 17, 0.2 + 2n*dist.
        With space_controls_speed_limit_offset it measures from the docking
        radius instead, matching DRL for Space Controls (2024) Eq. 5, bounded
        so a deputy inside the radius never drops below 0.2 m/s.

        Args:
            distance: Distance from the chief, meters.

        Returns:
            The speed limit at that distance, m/s.
        """
        if self.space_controls_speed_limit_offset:
            distance = max(0.0, distance - self.dock_dist)
        return 0.2 + 2 * self.n * distance

    def stopping_distance(self, speed: float) -> float:
        """Constant-deceleration stopping distance at max thrust: d = v^2/(2*a_max).

        Same formula as docking_env.py's. Used by the optional braking
        margin observation and the optional proximity-scaled velocity
        penalty; neither is active by default.

        Args:
            speed: Current speed in m/s.

        Returns:
            Required stopping distance in meters.
        """
        a_max = self.u_max / self.m  # max acceleration per axis (m/s^2)
        return speed ** 2 / (2 * a_max)

    def _get_obs(self) -> np.ndarray:
        """Build the observation for the current state.

        By default the 7-element state, timestep normalized by
        obs_time_norm, which defaults to max_episode_len (matching
        docking_env.py). Flags extend it: saferl_obs appends
        [speed, max_vel_limit], custom_braking_margin_obs appends the
        braking margin, normalize_obs also scales positions/velocities
        toward [-1, 1], and space_controls_obs drops the timestep instead.

        Returns:
            The observation array, 6 to 10 elements depending on flags.
        """
        obs = np.copy(self.state)
        obs[6] = obs[6] / self.obs_time_norm
        if self.normalize_obs:
            obs[0:3] = obs[0:3] / OBS_POS_NORM
            obs[3:6] = obs[3:6] / OBS_VEL_NORM
        if self.space_controls_obs:
            obs = obs[0:6]  # drop the timestep counter

        # speed/max_vel_limit and the braking margin are computed from
        # the raw (unnormalized) state.
        distance = vec_norm(self.state[0:3])
        speed = vec_norm(self.state[3:6])
        if self.saferl_obs:
            max_vel_limit = self.velocity_limit(distance)
            obs = np.concatenate([obs, np.array([speed, max_vel_limit])])
        if self.custom_braking_margin_obs:
            margin = distance - self.stopping_distance(speed)
            if self.normalize_obs:
                margin = margin / OBS_POS_NORM
            obs = np.concatenate([obs, np.array([margin])])
        return obs

    def sample_state_space(self) -> np.ndarray:
        """Sample a valid random starting state.

        Samples position and velocity uniformly, then rejects and
        resamples if the position is out of the allowed range or the
        initial speed exceeds the safety speed limit.

        Returns:
            A valid 7-element starting state with timestep counter set to 0.

        Raises:
            RuntimeError: If no valid state is found within
                MAX_START_SAMPLE_TRIES, meaning the bounds are too tight.
        """
        low = np.array([-self.max_start_dist] * 3 + [-self.max_start_speed] * 3)
        high = np.array([self.max_start_dist] * 3 + [self.max_start_speed] * 3)
        # Loop, not recursion, to avoid a stack overflow on very tight bounds.
        for _ in range(MAX_START_SAMPLE_TRIES):
            # Uses the env's own seeded generator, not Box.sample(), since
            # Box can't reuse an external generator in this Gymnasium version.
            sampled_state = self.np_random.uniform(low=low, high=high)
            rel_dist = vec_norm(sampled_state[0:3])
            rel_speed = vec_norm(sampled_state[3:6])
            # space_controls_init_vel_scale is 0.8 with the flag on, else 1.0.
            if ((rel_dist < self.min_start_dist) or
                (rel_dist > self.max_start_dist) or
                (rel_speed > self.space_controls_init_vel_scale * self.velocity_limit(rel_dist))):
                continue
            return np.concatenate([sampled_state, np.array([0])])
        raise RuntimeError(
            f"Could not draw a valid start state in {MAX_START_SAMPLE_TRIES} tries "
            f"(distance {self.min_start_dist} to {self.max_start_dist} m, per-axis "
            f"speed up to {self.max_start_speed} m/s). The accepted region is too "
            f"small for rejection sampling; widen the distance range or the speed bound."
        )

    def step(self, action: np.ndarray, drift=False) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment one step.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.
            drift: True to advance by drift_step_len (a coast step)
                instead of the normal step_len.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        self.time_step += 1
        old_state = np.copy(self.state)  # deep copy
        self.state = self.propagate(action, drift)
        # fuel_used: L2 norm, what one gimbaled thruster would have spent. Secondary metric.
        self.fuel_used += vec_norm(action) / self.m * self.step_len
        # fuel_used_l1: L1 norm, proportional to propellant
        # used. Primary metric; drives the fuel penalty and max_dv budget.
        self.step_delta_v = float(np.sum(np.abs(action))) / self.m * self.step_len
        self.fuel_used_l1 += self.step_delta_v
        reward, terminated, truncated = self.rewards(old_state)
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def propagate(self, action: np.ndarray, drift: bool = False) -> np.ndarray:
        """Integrate the CWH dynamics forward one step.

        Uses RK45 with tighter tolerances for numerical stability.
        Falls back to the current state if integration fails.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.
            drift: True to integrate over drift_step_len (a coast step)
                instead of step_len.

        Returns:
            New 7-element state after one step.
        """
        dt = self.step_len if not drift else self.drift_step_len

        if FAST_ANALYTIC_PROPAGATION:
            # Exact solution, so there is no integrator to fail here.
            transition, input_matrix = propagation_matrices(self.n, self.m, dt)
            new_state = transition @ self.state[0:6] + input_matrix @ np.asarray(
                action, dtype=float)
            return np.concatenate([new_state, np.array([1 + self.state[6]])])

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

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute the state derivative using CWH equations.

        Args:
            t: Current time (required by solve_ivp, not used directly).
            x: Current 6-element state [x, y, z, vx, vy, vz].
            u: Control input [Fx, Fy, Fz] in Newtons.

        Returns:
            State derivative as a 6-element array.
        """
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
        """Compute the reward and termination flags for the current step.

        By default a speed violation ends the episode immediately with a
        flat -1. With saferl_vel_constraint on, violations instead
        accumulate a graded penalty and only end the episode once
        saferl_vel_budget is exhausted.

        Args:
            last_state: The state from before this step, used to compute
                the change in distance.

        Returns:
            A tuple of (reward, terminated, truncated).
        """
        tot_step_rew = 0

        docked = self.is_docked()
        crashed = self.is_crashed()
        out_of_time = self.is_out_of_time()
        out_of_fuel = self.is_out_of_fuel()
        out_of_bounds = self.is_out_of_bounds()
        unsafe = self.is_unsafe()

        current_distance = float(vec_norm(self.state[0:3]))
        current_speed = float(vec_norm(self.state[3:6]))
        speed_limit = self.velocity_limit(current_distance)

        # Either constraint form counts as "on" here. They differ in the
        # bias term (see __init__) and whether the budget can end the
        # episode (see vel_budget_exhausted).
        vel_constraint_active = self.saferl_vel_constraint or self.space_controls_vel_constraint

        # Accumulate the graded penalty while violating, regardless of reward
        # structure, so budget behavior matches in dense and sparse modes.
        violation_penalty = 0.0
        if vel_constraint_active and unsafe:
            violation_penalty = (
                self.saferl_vel_violation_scale * (current_speed - speed_limit)
                + self.saferl_vel_violation_bias
            )
            self.vel_violation_reward_sum += violation_penalty

        # With either constraint form on, only exhausting the budget ends the
        # episode; under space_controls_vel_constraint it never exhausts
        # (see vel_budget_exhausted).
        unsafe_ends_episode = (
            self.vel_budget_exhausted() if vel_constraint_active else unsafe
        )

        term = (docked or crashed or out_of_bounds or out_of_fuel or
                unsafe_ends_episode or out_of_time)
        trunc = False

        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0
        if self.reward_structure == "dense":  # TODO det when training refiner
            if docked:
                if self.saferl_success_reward and self.saferl_success_time_bonus:
                    # SafeRL (2022) Table 8's coupled formula: +2 - t/t_max.
                    # (A single formula, not flat +2 reward with time bonus added separately.)
                    tot_step_rew += (
                        SAFERL_SUCCESS_REWARD_VALUE
                        - self.state[6] / self.max_episode_len)
                else:
                    tot_step_rew += (
                        SAFERL_SUCCESS_REWARD_VALUE if self.saferl_success_reward else 1)
                    tot_step_rew += self.success_time_bonus()
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time:
                tot_step_rew += -1
            elif unsafe and not vel_constraint_active:
                # paper_unsafe_reward overrides: 0 (paper) or -1 (default).
                tot_step_rew += 0.0 if self.paper_unsafe_reward else -1.0
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
            tot_step_rew += (prox_penalty + self.time_penalty + violation_penalty)
            if self.custom_vel_penalty:
                # Penalize speed more near the target; 0.3 keeps it active far away.
                braking_zone = max(self.dock_dist, self.stopping_distance(current_speed))
                approach_factor = max(0.3, min(1.0, braking_zone / max(current_distance, 1.0)))
                tot_step_rew += (self.vel_penalty_coeff
                                 * max(current_speed - speed_limit, 0)
                                 * approach_factor)
            if self.saferl_delta_v_penalty:
                # SafeRL Delta-v fuel penalty (scale * step Delta-v).
                tot_step_rew += -self.saferl_delta_v_scale * self.step_delta_v
        return tot_step_rew, term, trunc

    def vel_budget_exhausted(self) -> bool:
        """Check whether the velocity-violation budget is used up.

        Always False if saferl_vel_constraint is off or space_controls_vel_constraint is on.
        space_controls > saferl in priority.

        Returns:
            True once the accumulated penalties reach saferl_vel_budget.
        """
        return (self.saferl_vel_constraint and
                not self.space_controls_vel_constraint and
                self.vel_violation_reward_sum <= self.saferl_vel_budget)

    def is_docked(self) -> bool:
        """Check if the deputy has successfully docked.

        Returns:
            True if within docking distance and speed thresholds.
        """
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) < self.dock_speed)

    def success_time_bonus(self) -> float:
        """Bonus for winning early, 0 when saferl_success_time_bonus is off.

        Scales from +1 for an instant win down to 0 at the episode
        limit, matching docking_env.py's version. Bounded at 0 so a win
        at or past the limit never subtracts reward.

        Scored at the current step for both win paths (direct dock and
        coast win), not the coast's predicted finish time, since a coast
        often runs longer than the stage's step budget.

        rewards() skips this method on the direct-dock path when
        saferl_success_reward is on, using SafeRL (2022) Table 8's
        coupled formula (+2 - t/t_max) inline instead.

        Returns:
            The bonus to add to the win reward.
        """
        if not self.saferl_success_time_bonus:
            return 0.0
        return max(0.0, 1 - self.state[6] / self.max_episode_len)

    def is_crashed(self) -> bool:
        """Check if the deputy hit the chief too fast to dock safely.

        Returns:
            True if within docking distance but above the speed threshold.
        """
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) >= self.dock_speed)

    def is_out_of_fuel(self) -> bool:
        """Check if the deputy has run out of propellant.

        Budgets against fuel_used_l1, not fuel_used, because propellant
        used is the per-axis sum: each fixed thruster spends fuel
        proportional to its |thrust|. This matches the Delta-v
        reward penalty, which uses the L1 step_delta_v.

        Returns:
            True if cumulative Delta-v (L1) exceeds max_dv.
        """
        return self.fuel_used_l1 > self.max_dv

    def is_out_of_bounds(self) -> bool:
        """Check if the deputy has left the allowed region.

        Uses a sphere boundary.

        Returns:
            True if distance from the origin exceeds max_boundary_box.
        """
        return vec_norm(self.state[0:3]) > self.max_boundary_box

    def is_out_of_time(self) -> bool:
        """Check if the episode has run out of time.

        Returns:
            True once the step count reaches max_episode_len.
        """
        return self.state[6] >= self.max_episode_len  # TODO > or >=?

    def is_unsafe(self) -> bool:
        """Check if the deputy is above the distance-scaled speed limit.

        Returns:
            True if the current speed exceeds velocity_limit().
        """
        current_distance = float(vec_norm(self.state[0:3]))
        current_speed = float(vec_norm(self.state[3:6]))
        speed_limit = self.velocity_limit(current_distance)
        return current_speed - speed_limit > 0

    def _unsafe_is_terminal(self) -> bool:
        """Check whether the current speed state should end the episode.

        Matches rewards()'s termination rule; used by drift lookahead.

        Returns:
            True if the speed state is a terminal failure.
        """
        vel_constraint_active = self.saferl_vel_constraint or self.space_controls_vel_constraint
        return (self.vel_budget_exhausted() if vel_constraint_active
                else self.is_unsafe())

    def only_oot(self) -> bool:
        """Check if running out of time is the sole reason to end.

        Used by the drift wrappers to decide whether to still check for
        a coast opportunity when time runs out.

        Returns:
            True if time is the only ending condition currently met.
        """
        return (not (self.is_docked() or self.is_crashed() or
                     self.is_out_of_bounds() or self.is_out_of_fuel()
                     or self._unsafe_is_terminal())) and self.is_out_of_time()

    def term_other_than_oot(self) -> bool:
        """Check if the episode ended for any reason other than time.

        Used by the drift lookahead to detect a real ending mid-coast.

        Returns:
            True if a non-time ending condition is met.
        """
        return (self.is_docked() or self.is_crashed() or
                self.is_out_of_bounds() or self.is_out_of_fuel()
                or self._unsafe_is_terminal())

    def close(self) -> None:
        """Clean up environment resources. Required by Gymnasium interface."""
        pass


def _det_drift(env: "DriftDockingEnv3D") -> tuple[bool, int]:
    """Check whether coasting from here would reach the dock.

    Deep-copies the env and simulates zero thrust forward until
    something ends the episode or the lookahead budget runs out.
    Pure physics, no learned parameters.

    Shared by DriftTrainEnv.det_drift() and DriftTestEnv.det_drift()
    to avoid duplicate logic for this safety-critical function (decides
    whether an episode ends in a coast win).

    Args:
        env: The wrapped DriftDockingEnv3D instance (DriftTrainEnv.env
            or DriftTestEnv.env), not the wrapper itself.

    Returns:
        A tuple of (would_dock, steps), where steps is how many
        coasting steps the simulation took before it ended.
    """
    new_env = copy.deepcopy(env)
    t = 0
    for _ in range(env.lookahead_len):
        drift_action = np.array([0.0, 0.0, 0.0])
        t += 1
        new_env.step(drift_action, True)
        if new_env.term_other_than_oot():
            return bool(new_env.is_docked()), t
    return False, t


class DriftTrainEnv(gym.Env):
    """Wrapper for docking environment that looks ahead each time
    step to determine if docking condition can be achieved by drifting.
    """
    def __init__(self, **kwargs) -> None:
        """Build the wrapped environment.

        Args:
            **kwargs: Passed straight to DriftDockingEnv3D.
        """
        self.env = DriftDockingEnv3D(**kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the wrapped environment.

        Args:
            seed: Random seed for reproducibility.
            options: Unused, required by the Gymnasium interface.

        Returns:
            A tuple of (observation, info dict).
        """
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one step, ending the episode early on a coast win.

        If the lookahead says coasting from here would dock, the episode
        is credited with the coast reward and ends immediately.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
            info["coast_win"] is True when this step ended the episode by
            handing off to a coast, and False otherwise.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        # Recorded because term alone cannot distinguish a coast win from other
        # endings, and det_drift() (a deep-copy forward simulation) is expensive to rerun.
        info["coast_win"] = False
        if (not done) or (self.env.only_oot()):
            is_drift, the_time = self.det_drift()
            if is_drift:
                rew += 10
                # Scored at the decision step, not the coast's predicted finish,
                # since the coast often runs longer than the stage's step budget.
                rew += self.env.success_time_bonus()
                term = True
                info["coast_win"] = True
        return obs, rew, term, trunc, info

    def det_drift(self) -> tuple[bool, int]:
        """Check whether coasting from here would reach the dock.

        Uses _det_drift() while keeping the self.det_drift() interface.

        Returns:
            A tuple of (would_dock, steps), where steps is how many
            coasting steps the simulation took before it ended.
        """
        return _det_drift(self.env)


class DriftTestEnv(gym.Env):
    """
    Wrapper for docking environment that looks ahead each time step to
    determine if docking condition can be achieved by drifting.
    """
    def __init__(self, **kwargs) -> None:
        """Build the wrapped environment.

        Args:
            **kwargs: Passed straight to DriftDockingEnv3D.
        """
        self.env = DriftDockingEnv3D(**kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.is_drifting = False

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the wrapped environment and clear the drift flag.

        Args:
            seed: Random seed for reproducibility.
            options: Unused, required by the Gymnasium interface.

        Returns:
            A tuple of (observation, info dict).
        """
        self.is_drifting = False
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one step, recording whether a coast win is available.

        Unlike DriftTrainEnv, this does not end the episode or add
        reward on a coast win; it only sets is_drifting so the eval
        scripts can see the agent has handed off to a coast.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        if PRINT_WIN_MESSAGE and self.env.is_docked():
            print('WIN!')
        if (not done) or (self.env.only_oot()):
            if not self.is_drifting:
                is_drift, the_time = self.det_drift()
                if is_drift:
                    self.is_drifting = True
        return obs, rew, term, trunc, info

    def det_drift(self) -> tuple[bool, int]:
        """Check whether coasting from here would reach the dock.

        See _det_drift() for the logic; this wrapper keeps
        self.det_drift() available to callers.

        Returns:
            A tuple of (would_dock, steps), where steps is how many
            coasting steps the simulation took before it ended.
        """
        return _det_drift(self.env)
