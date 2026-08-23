"""Spacecraft docking environment for direct-control (no drift/coast) agents.

Uses Clohessy-Wiltshire-Hill orbital dynamics with a configurable reward
structure. Supports optional flags to replicate SafeRL and Space Controls
baseline configurations for comparison.

Used by nodrift_initial_trainer.py and the nodrift evaluation scripts.
"""

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

# Prefix explains the source: SAFERL_ from SafeRL (2022), SPACE_CONTROLS_
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
# using it. Every curriculum stage supplies pos_thresh /
# max_boundary_box / max_control directly.

# True: dock at DRL for Space Controls (2024)'s 10 m radius.
# False: use pos_thresh, normally 0.5 m (SafeRL 2022 Table 4).
SPACE_CONTROLS_DOCK_RADIUS = False
SPACE_CONTROLS_DOCK_RADIUS_VALUE = 10.0

# True: treat an out-of-bounds condition as the Space Controls paper's
# 800 m limit (a looser safety boundary).
# False: use the environment's default max_boundary_box (usually 200 m).
SPACE_CONTROLS_MAX_DISTANCE = False
SPACE_CONTROLS_MAX_DISTANCE_VALUE = 800.0

# True: use the SafeRL 2022 runaway-check limit of 40,000 m.
# False: keep the default 200 m boundary. If both this flag and
# SPACE_CONTROLS_MAX_DISTANCE are enabled, this one takes precedence.
SAFERL_MAX_DISTANCE = False
SAFERL_MAX_DISTANCE_VALUE = 40_000.0

# True: use the low-thrust control setting from the Space Controls 2024 sweep:
# 0.1 N per axis.
# False: use the normal per-axis limit (usually 1.0 N).
SPACE_CONTROLS_LOW_THRUST = False
SPACE_CONTROLS_LOW_THRUST_VALUE = 0.1

# True: add a "brake now" signal to the observation: distance minus the
# stopping distance. This is a custom feature, not part of the original SafeRL
# setup.
# False: leave the observation unchanged.
CUSTOM_BRAKING_MARGIN_OBS = False

# True: scale observations to a simpler normalized form:
# position values are divided by 100, velocity values by 0.5.
# False: use raw values directly.
# For a SafeRL-style setup, keep this enabled.
NORMALIZE_OBS = False
OBS_POS_NORM = 100.0
OBS_VEL_NORM = 0.5

# Maximum number of times to try generating a valid starting state before
# giving up. If this is reached, the start-state bounds are probably too tight.
MAX_START_SAMPLE_TRIES = 10000

# Time penalty applied each step as a total penalty spread over the episode,
# with a minimum floor of TIME_PENALTY_FLOOR.
TIME_PENALTY_TOTAL = -1.0
TIME_PENALTY_FLOOR = -0.005

# True: print "WIN!" when a docking attempt succeeds.
# False: stay silent; evaluation scripts usually print their own pass/fail
# messages.
PRINT_WIN_MESSAGE = False

# True: use the exact closed-form Clohessy-Wiltshire-Hill solution instead of
# the numerical solve_ivp solver.
# False: use solve_ivp (RK45), which is the original implementation.
# The two methods match to about 1e-11; this is mainly a performance option.
FAST_ANALYTIC_PROPAGATION = False


class SpaceCraftDockingEnv3D(gym.Env):
    """Spacecraft docking environment using Clohessy-Wiltshire-Hill (CWH) dynamics.

    The deputy spacecraft tries to dock with the chief spacecraft at the
    origin. No drift mechanic is included; this environment is for
    training a direct-docking agent.

    State vector (7 elements): [x, y, z, vx, vy, vz, timestep_counter]

    Attributes:
        fixed_start (bool): If True, always start from fixed_state.
        fixed_state (numpy.ndarray): Starting state used when fixed_start is True.
        reward_structure (str): "dense" for shaped reward, "sparse" for win-only.
        max_episode_len (int): Maximum number of steps before the episode ends.
        max_boundary_box (float): Maximum distance from origin before out-of-bounds.
        u_max (float): Maximum thrust per axis in Newtons.
        max_dv (float): Maximum propellant budget, compared against
            fuel_used_l1 (see is_out_of_fuel).
        dock_dist (float): Position threshold for successful docking in meters.
        dock_speed (float): Speed threshold for successful docking in m/s.
        min_start_dist (float): Minimum starting distance from origin in meters.
        max_start_dist (float): Maximum starting distance from origin in meters.
        max_start_speed (float): Maximum starting speed in m/s.
        step_len (int): Duration of each step in seconds.
        fuel_used (float): Cumulative L2 Delta-v this episode, what a
            single gimbaled thruster would have spent to fly the same
            trajectory. Reported as a secondary metric.
        fuel_used_l1 (float): Cumulative L1 Delta-v, proportional to
            propellant used by the three fixed per-axis
            thrusters. The primary fuel metric, and what the Delta-v
            reward penalty and the max_dv budget both use.
        time_step (int): Step counter, also stored in state[6].
        n (float): Mean motion of the chief orbit in rad/s.
        m (float): Mass of the deputy spacecraft in kg.
        time_penalty (float): Per-step time penalty. By default
            TIME_PENALTY_TOTAL spread evenly over max_episode_len
            (floored, so it varies with episode length); can be set to an
            explicit constant via the time_penalty constructor argument.
        dist_coeff (float): Reward coefficient for change in distance each step.
        vel_penalty_coeff (float): Reward coefficient for exceeding the speed limit.
        normalize_obs (bool): If True, statically normalize positions and
            velocities in the observation (SafeRL constants).
        vel_violation_reward_sum (float): Velocity-violation penalties
            accumulated this episode; episode fails at saferl_vel_budget.
    """

    def __init__(self,
                 fixed_start=False,
                 fixed_state=np.array([100, 0, 0, 0, 0, 0, 0]),
                 reward_structure="dense",
                 max_episode_len=25,
                 max_boundary_box=10.0,
                 max_control=1.0,
                 max_total_dv=2_500,
                 pos_thresh=0.5,
                 speed_thresh=0.2,
                 min_init_pos_bound=0.5,
                 max_init_pos_bound=1,
                 max_init_vel_bound=0.1,
                 step_len=1,
                 fuel_used=None,
                 time_step=None,
                 saferl_obs=SAFERL_OBS,
                 saferl_exp_dist_reward=SAFERL_EXP_DIST_REWARD,
                 saferl_delta_v_penalty=SAFERL_DELTA_V_PENALTY,
                 saferl_success_time_bonus=SAFERL_SUCCESS_TIME_BONUS,
                 saferl_success_reward=SAFERL_SUCCESS_REWARD,
                 saferl_vel_constraint=SAFERL_VEL_CONSTRAINT,
                 saferl_max_distance=SAFERL_MAX_DISTANCE,
                 saferl_no_time_penalty=SAFERL_NO_TIME_PENALTY,
                 custom_braking_margin_obs=CUSTOM_BRAKING_MARGIN_OBS,
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
                 time_penalty=None):
        self.fixed_start = fixed_start
        self.fixed_state = fixed_state
        self.reward_structure = reward_structure
        self.max_episode_len = max_episode_len
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
        self.fuel_used = fuel_used
        self.fuel_used_l1 = 0.0
        self.time_step = time_step

        self.n = 0.001027 # Mean motion of chief orbit (rad/s)
        self.m = 12       # Deputy mass (kg)
        # Default: TIME_PENALTY_TOTAL / max_episode_len (floored).
        # Override order: time_penalty > space_controls > paper > saferl_no_time_penalty.
        if time_penalty is not None:
            self.time_penalty = time_penalty
        elif space_controls_time_penalty:
            self.time_penalty = SPACE_CONTROLS_TIME_PENALTY_VALUE
        elif paper_time_penalty:
            self.time_penalty = PAPER_TIME_PENALTY_VALUE
        elif saferl_no_time_penalty:
            self.time_penalty = 0.0
        else:
            self.time_penalty = max(TIME_PENALTY_TOTAL / self.max_episode_len, TIME_PENALTY_FLOOR)
        # Matches the paper's stated coefficient and drift_env.py's
        # dist_coeff, so the two envs share the same closing-distance pull.
        self.dist_coeff = -0.0005  # approach-reward coefficient
        self.vel_penalty_coeff = -0.02  # speed-limit-violation penalty

        # SafeRL comparison options (see constants for details).
        self.saferl_obs = saferl_obs
        self.saferl_exp_dist_reward = saferl_exp_dist_reward
        self.saferl_delta_v_penalty = saferl_delta_v_penalty
        self.saferl_success_time_bonus = saferl_success_time_bonus
        self.saferl_success_reward = saferl_success_reward
        self.saferl_vel_constraint = saferl_vel_constraint
        self.custom_braking_margin_obs = custom_braking_margin_obs
        self.normalize_obs = normalize_obs
        # Space Controls comparison options (see constants for details).
        self.space_controls_obs = space_controls_obs
        self.space_controls_init_vel_scale = SPACE_CONTROLS_INIT_VEL_SCALE if space_controls_init_vel_fraction else 1.0
        self.space_controls_speed_limit_offset = space_controls_speed_limit_offset
        self.space_controls_vel_constraint = space_controls_vel_constraint
        self.saferl_dist_scale = SAFERL_DIST_SCALE
        self.saferl_dist_a = np.log(2.0) / SAFERL_DIST_PIVOT  # closing reward doubles every pivot m
        self.saferl_delta_v_scale = SAFERL_DELTA_V_SCALE
        self.saferl_vel_violation_scale = SAFERL_VEL_VIOLATION_SCALE
        # space_controls_vel_constraint drops the flat per-violating-step
        # term; saferl_vel_constraint keeps it. Space Controls takes
        # priority when both are on.
        self.saferl_vel_violation_bias = (
            0.0 if space_controls_vel_constraint else SAFERL_VEL_VIOLATION_BIAS)
        self.saferl_vel_budget = SAFERL_VEL_BUDGET
        self.step_delta_v = 0.0  # Delta-v of the most recent step, set in step()
        self.vel_violation_reward_sum = 0.0  # Constraint budget used this episode

        self.action_space = spaces.Box(
            low=np.array([-self.u_max] * 3, dtype=np.float32),
            high=np.array([self.u_max] * 3, dtype=np.float32)
        )
        # 7 elements by default (6 with space_controls_obs), +2 for saferl_obs,
        # +1 for custom_braking_margin_obs. Unbounded; out-of-bounds terminates rather than clips.
        obs_dim = 6 if self.space_controls_obs else 7
        if self.saferl_obs:
            obs_dim += 2
        if self.custom_braking_margin_obs:
            obs_dim += 1
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim, dtype=np.float32),
            high=np.array([np.inf] * obs_dim, dtype=np.float32)
        )
        self.state = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to a starting state.

        Args:
            seed: Random seed for reproducibility.
            options: Unused, required by Gymnasium interface.

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
        max_episode_len. Flags extend it: saferl_obs appends
        [speed, max_vel_limit], custom_braking_margin_obs appends the
        braking margin, normalize_obs also scales positions/velocities
        toward [-1, 1], and space_controls_obs drops the timestep instead.

        Returns:
            The observation array, 6 to 10 elements depending on flags.
        """
        obs = np.copy(self.state)
        obs[6] = obs[6] / self.max_episode_len
        if self.normalize_obs:
            obs[0:3] = obs[0:3] / OBS_POS_NORM
            obs[3:6] = obs[3:6] / OBS_VEL_NORM
        if self.space_controls_obs:
            obs = obs[0:6]  # drop the timestep counter

        if not self.saferl_obs and not self.custom_braking_margin_obs:
            return obs

        distance = vec_norm(self.state[0:3])
        speed = vec_norm(self.state[3:6])
        if self.saferl_obs:
            # speed and max_vel_limit stay unscaled even with normalize_obs on.
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
        """
        low = np.array([-self.max_start_dist] * 3 + [-self.max_start_speed] * 3)
        high = np.array([self.max_start_dist] * 3 + [self.max_start_speed] * 3)
        # Loop, not recursion, to avoid a stack overflow on very tight bounds.
        for _ in range(MAX_START_SAMPLE_TRIES):
            # Uses the env's seeded generator, not Box.sample(), since
            # Box can't reuse an external generator in this Gymnasium version.
            sampled_state = self.np_random.uniform(low=low, high=high)
            rel_dist = vec_norm(sampled_state[0:3])
            rel_speed = vec_norm(sampled_state[3:6])
            # Reject if too close, too far, or too fast. space_controls_init_vel_scale
            # is 0.8 with the flag on, else 1.0.
            if (rel_dist < self.min_start_dist or
                    rel_dist > self.max_start_dist or
                    rel_speed > self.space_controls_init_vel_scale * self.velocity_limit(rel_dist)):
                continue
            return np.concatenate([sampled_state, np.array([0])])  # append timestep counter
        raise RuntimeError(
            f"Could not draw a valid start state in {MAX_START_SAMPLE_TRIES} tries "
            f"(distance {self.min_start_dist} to {self.max_start_dist} m, per-axis "
            f"speed up to {self.max_start_speed} m/s). The accepted region is too "
            f"small for rejection sampling; widen the distance range or the speed bound."
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment one step.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        self.time_step += 1
        old_state = np.copy(self.state)
        self.state = self.propagate(action)
        # fuel_used: L2 norm, what one gimbaled thruster would have spent. Secondary metric.
        self.fuel_used += vec_norm(action) / self.m * self.step_len
        # step_delta_v / fuel_used_l1: L1 norm, proportional to propellant
        # used. Primary metric; drives the fuel penalty and max_dv budget.
        self.step_delta_v = float(np.sum(np.abs(action))) / self.m * self.step_len
        self.fuel_used_l1 += self.step_delta_v
        reward, terminated, truncated = self.rewards(old_state)
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def propagate(self, action: np.ndarray) -> np.ndarray:
        """Integrate the CWH dynamics forward one step.

        Uses RK45 with tighter tolerances for numerical stability.
        Falls back to the current state if integration fails.

        Args:
            action: Thrust vector [Fx, Fy, Fz] in Newtons.

        Returns:
            New 7-element state after one step.
        """
        if FAST_ANALYTIC_PROPAGATION:
            # Exact solution, so there is no integrator to fail here.
            transition, input_matrix = propagation_matrices(self.n, self.m, self.step_len)
            new_state = transition @ self.state[0:6] + input_matrix @ np.asarray(
                action, dtype=float)
            return np.concatenate([new_state, np.array([1 + self.state[6]])])

        t_span = (self.time_step, self.time_step + self.step_len)
        result = solve_ivp(
            self.dynamics,
            t_span,
            self.state[0:6],   # only pass position/velocity, not the timestep counter
            args=(action,),
            method='RK45',
            rtol=1e-5,
            atol=1e-8
        )
        if not result.success:
            return self.state  # hold current state if integration fails
        new_state = result.y[:, -1]
        # Increment the timestep counter stored in state[6].
        return np.concatenate([new_state, np.array([1 + self.state[6]])])

    def dynamics(self, t, x, u) -> np.ndarray:
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
                      [0, 0, -(self.n ** 2), 0, 0, 0]])
        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1 / self.m, 0, 0],
                      [0, 1 / self.m, 0],
                      [0, 0, 1 / self.m]])
        return A.dot(x) + B.dot(u)

    def rewards(self, last_state: np.ndarray) -> tuple[float, bool, bool]:
        """Compute the reward and termination flags for the current step.

        A soft, proximity-scaled speed penalty always applies. With
        saferl_vel_constraint on, violations also accumulate a graded
        penalty, and the episode fails once that sum reaches
        saferl_vel_budget.

        Args:
            last_state: The state from before this step, used to compute
                the change in distance.

        Returns:
            A tuple of (reward, terminated, truncated).
        """
        tot_step_rew = 0

        current_distance = float(vec_norm(self.state[0:3]))
        current_speed = float(vec_norm(self.state[3:6]))
        speed_limit = self.velocity_limit(current_distance)

        docked = self.is_docked()
        crashed = self.is_crashed()
        out_of_time = self.is_out_of_time()
        out_of_fuel = self.is_out_of_fuel()
        out_of_bounds = self.is_out_of_bounds()

        # Accumulate the graded penalty while violating, regardless of reward
        # structure, so budget termination behaves the same in dense and sparse modes.
        violation_penalty = 0.0
        vel_constraint_active = self.saferl_vel_constraint or self.space_controls_vel_constraint
        if vel_constraint_active and current_speed > speed_limit:
            violation_penalty = (
                self.saferl_vel_violation_scale * (current_speed - speed_limit)
                + self.saferl_vel_violation_bias
            )
            self.vel_violation_reward_sum += violation_penalty

        term = (docked or crashed or out_of_bounds or out_of_fuel or
                self.vel_budget_exhausted())
        # Out-of-time is truncation, not termination, to preserve the
        # episode value estimate in PPO.
        trunc = False if term else out_of_time

        if PRINT_WIN_MESSAGE and docked:
            print("WIN!")

        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0

        if self.reward_structure == "dense":
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
                    if self.saferl_success_time_bonus:
                        # Up to +1 for docking early, bounded at 0 so a late win
                        # never subtracts reward. Matches drift_env's success_time_bonus().
                        tot_step_rew += max(
                            0.0, 1 - self.state[6] / self.max_episode_len)
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time:
                tot_step_rew += -1
            # Budget exhaustion gets no terminal adjustment here; the
            # accumulated violation penalties are the entire cost.

            prev_distance = float(vec_norm(last_state[0:3]))
            if self.saferl_exp_dist_reward:
                # SafeRL exponential distance-change reward (stronger near target).
                approach_reward = self.saferl_dist_scale * (
                    np.exp(-self.saferl_dist_a * current_distance)
                    - np.exp(-self.saferl_dist_a * prev_distance)
                )
            else:
                # Reward closing distance. dist_coeff is negative, so the
                # delta must be (current - prev) to reward getting closer.
                approach_reward = self.dist_coeff * (current_distance - prev_distance)
            # Speed penalty scaled by proximity: fast is fine far away, penalized
            # near target. 0.3 floor keeps the penalty from vanishing at long range.
            braking_zone = max(self.dock_dist, self.stopping_distance(current_speed))
            approach_factor = max(0.3, min(1.0, braking_zone / max(current_distance, 1.0)))
            vel_penalty = self.vel_penalty_coeff * max(current_speed - speed_limit, 0) * approach_factor
            tot_step_rew += vel_penalty + approach_reward + self.time_penalty + violation_penalty
            if self.saferl_delta_v_penalty:
                tot_step_rew += -self.saferl_delta_v_scale * self.step_delta_v

        return tot_step_rew, term, trunc

    def is_docked(self) -> bool:
        """Check if the deputy has successfully docked.

        Returns:
            True if within docking distance and speed thresholds.
        """
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) < self.dock_speed)

    def is_crashed(self) -> bool:
        """Check if the deputy hit the chief too fast to dock safely.

        Returns:
            True if within docking distance but above the speed threshold.
        """
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) >= self.dock_speed)

    def is_unsafe(self) -> bool:
        """Check if the deputy is above the distance-scaled speed limit.

        When saferl_vel_constraint is on, violating steps also add to
        vel_violation_reward_sum in rewards(); otherwise only the soft
        proximity-scaled penalty applies.

        Returns:
            True if the current speed exceeds velocity_limit().
        """
        current_distance = float(vec_norm(self.state[0:3]))
        current_speed = float(vec_norm(self.state[3:6]))
        return current_speed > self.velocity_limit(current_distance)

    def vel_budget_exhausted(self) -> bool:
        """Check whether the velocity-violation budget is used up.

        The terminal failure condition of SafeRL (2022) Table 10's
        budgeted constraint. Always False under
        space_controls_vel_constraint: DRL for Space Controls (2024)
        penalizes violations but never terminates on them, and that flag
        takes priority over saferl_vel_constraint when both are on
        (see __init__).

        Returns:
            True once the accumulated penalties reach saferl_vel_budget.
        """
        return (self.saferl_vel_constraint and
                not self.space_controls_vel_constraint and
                self.vel_violation_reward_sum <= self.saferl_vel_budget)

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
        return self.state[6] >= self.max_episode_len  # TODO > or >=

    def only_oot(self) -> bool:
        """Check if running out of time is the sole reason to end.

        Used by the drift wrappers to decide whether to still check for
        a coast opportunity when time runs out.

        Returns:
            True if time is the only ending condition currently met.
        """
        return (not (self.is_docked() or self.is_crashed() or
                     self.is_out_of_bounds() or self.is_out_of_fuel())) \
               and self.is_out_of_time()

    def term_other_than_oot(self) -> bool:
        """Check if the episode ended for any reason other than time.

        Used by the drift lookahead to detect a real ending mid-coast.

        Returns:
            True if a non-time ending condition is met.
        """
        return (self.is_docked() or self.is_crashed() or
                self.is_out_of_bounds() or self.is_out_of_fuel())

    def close(self) -> None:
        """Clean up environment resources. Required by Gymnasium interface."""
        pass
