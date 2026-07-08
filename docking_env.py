"""
docking_env.py

Direct-docking environment: CWH dynamics and no drift mechanic. 
Dense reward with a proximity-scaled velocity penalty, plus 
optional SafeRL comparison flags (obs layout, reward shape, 
fuel penalty, budgeted velocity constraint, static observation 
normalization).

Used by nodrift_initial_trainer.py and the nodrift eval scripts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import norm as vec_norm


# ===================== SafeRL comparison options =====================
# Optional switches to mirror the AFRL/ACT3 SafeRL docking baseline (IEEE
# Aerospace 2022) for A/B comparison. Usually defaults to False (current
# drifter behavior); flip here or via the matching constructor argument.
# SafeRL references: saferl/aerospace/tasks/docking/processors.py and
# saferl/environment/tasks/processor/reward.py.

# True: append [speed, max_vel_limit] to the obs (SafeRL DockingObservation).
# False: 7-element obs [x, y, z, vx, vy, vz, timestep].
SAFERL_OBS = True

# True: exponential distance-change reward (gradient strongest near target).
# False: linear approach reward.
SAFERL_EXP_DIST_REWARD = True
SAFERL_DIST_PIVOT = 100.0   # SafeRL 'pivot' (m): closing reward doubles every pivot
SAFERL_DIST_SCALE = 2.0     # SafeRL 'c'

# True: subtract a delta-V fuel penalty each step. False: no fuel term.
SAFERL_DELTA_V_PENALTY = False
SAFERL_DELTA_V_SCALE = 0.01  # SafeRL ProportionalRewardProcessor 'scale' on delta_v

# True: on a successful dock, add a time bonus of 1 - timestep /
# max_episode_len (SafeRL SuccessRewardProcessor), so docking with time
# to spare pays up to +1 extra. False: flat +1 dock reward only.
# Off by default: it adds a second objective (finish fast) that could
# conflict with fuel economy, so enable it deliberately as an ablation.
SAFERL_SUCCESS_TIME_BONUS = False

# True: enforce SafeRL's budgeted velocity constraint. Each step over the
# speed limit pays a graded penalty (SCALE * violation + BIAS) into a
# per-episode sum; the episode fails once that sum reaches BUDGET. SafeRL
# never ends an episode on a first mid-flight violation, so this replaces
# the old instant-termination behavior.
# False: soft proximity-scaled speed penalty only, no budget or failure.
SAFERL_VEL_CONSTRAINT = True
SAFERL_VEL_VIOLATION_SCALE = -0.01  # SafeRL ProportionalRewardProcessor 'scale'
SAFERL_VEL_VIOLATION_BIAS = -0.01   # SafeRL 'bias', flat cost per violating step
SAFERL_VEL_BUDGET = -5.0            # SafeRL 'lower_bound'; fail at this sum

# True: append a braking margin (distance minus stopping distance) to the
# obs (a direct "start braking now" signal). False: obs size unchanged.
# Off by default: other scripts auto-detect obs layout by checking for
# exactly 9 elements (SAFERL_OBS), so this needs those checks updated too.
SAFERL_BRAKING_MARGIN_OBS = False

# True: statically normalize the observation like SafeRL (positions / 100,
# velocities / 0.5) so network inputs land near [-1, 1]. False: raw values.
# Earlier runs were trained on raw positions/velocities; set False to
# evaluate those checkpoints. The timestep is now always normalized.
NORMALIZE_OBS = True
OBS_POS_NORM = 100.0  # SafeRL DockingObservationProcessor position divisor
OBS_VEL_NORM = 0.5    # SafeRL velocity divisor

# Whole-episode time penalty, spread evenly across max_episode_len steps.
# A flat per-step value summed to -25 over a 5,000-step episode.
TIME_PENALTY_TOTAL = -1.0
# TIME_PENALTY_TOTAL is a per-step shaping cost
# SAFERL_SUCCESS_TIME_BONUS is a separate terminal reward for a successful dock.


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
        max_dv (float): Maximum total fuel budget in delta-V units.
        dock_dist (float): Position threshold for successful docking in meters.
        dock_speed (float): Speed threshold for successful docking in m/s.
        min_start_dist (float): Minimum starting distance from origin in meters.
        max_start_dist (float): Maximum starting distance from origin in meters.
        max_start_speed (float): Maximum starting speed in m/s.
        step_len (int): Duration of each step in seconds.
        fuel_used (float): Cumulative fuel used this episode.
        time_step (int): Step counter, also stored in state[6].
        n (float): Mean motion of the chief orbit in rad/s.
        m (float): Mass of the deputy spacecraft in kg.
        time_penalty (float): Per-step time penalty, TIME_PENALTY_TOTAL
            spread evenly over max_episode_len.
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
                 saferl_vel_constraint=SAFERL_VEL_CONSTRAINT,
                 saferl_braking_margin_obs=SAFERL_BRAKING_MARGIN_OBS,
                 normalize_obs=NORMALIZE_OBS):
        self.fixed_start = fixed_start
        self.fixed_state = fixed_state
        self.reward_structure = reward_structure
        self.max_episode_len = max_episode_len
        self.max_boundary_box = max_boundary_box
        self.u_max = max_control
        self.max_dv = max_total_dv
        self.dock_dist = pos_thresh
        self.dock_speed = speed_thresh
        self.min_start_dist = min_init_pos_bound
        self.max_start_dist = max_init_pos_bound
        self.max_start_speed = max_init_vel_bound
        self.step_len = step_len
        self.fuel_used = fuel_used
        self.time_step = time_step

        self.n = 0.001027 # Mean motion of chief orbit (rad/s)
        self.m = 12       # Deputy mass (kg)
        # Per-step time penalty scaled so the whole-episode total stays at
        # TIME_PENALTY_TOTAL, comparable to the +/-1 terminal rewards at
        # every stage regardless of episode length.
        self.time_penalty = TIME_PENALTY_TOTAL / self.max_episode_len
        self.dist_coeff = -0.01   # increased from -0.005 for stronger approach signal
        self.vel_penalty_coeff = -0.0075

        # SafeRL comparison options (see constants for details).
        self.saferl_obs = saferl_obs
        self.saferl_exp_dist_reward = saferl_exp_dist_reward
        self.saferl_delta_v_penalty = saferl_delta_v_penalty
        self.saferl_success_time_bonus = saferl_success_time_bonus
        self.saferl_vel_constraint = saferl_vel_constraint
        self.saferl_braking_margin_obs = saferl_braking_margin_obs
        self.normalize_obs = normalize_obs
        self.saferl_dist_scale = SAFERL_DIST_SCALE
        self.saferl_dist_a = np.log(2.0) / SAFERL_DIST_PIVOT  # closing reward doubles every pivot m
        self.saferl_delta_v_scale = SAFERL_DELTA_V_SCALE
        self.saferl_vel_violation_scale = SAFERL_VEL_VIOLATION_SCALE
        self.saferl_vel_violation_bias = SAFERL_VEL_VIOLATION_BIAS
        self.saferl_vel_budget = SAFERL_VEL_BUDGET
        self.step_delta_v = 0.0  # Delta-V of the most recent step, set in step()
        self.vel_violation_reward_sum = 0.0  # Constraint budget used this episode

        self.action_space = spaces.Box(
            low=np.array([-self.u_max] * 3),
            high=np.array([self.u_max] * 3)
        )
        # Observation is unbounded; the environment handles out-of-bounds
        # termination internally rather than clipping the observation.
        # 7 elements by default; +2 when saferl_obs appends [speed,
        # max_vel_limit]; +1 more when saferl_braking_margin_obs appends
        # the braking feature.
        obs_dim = 7
        if self.saferl_obs:
            obs_dim += 2
        if self.saferl_braking_margin_obs:
            obs_dim += 1
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim),
            high=np.array([np.inf] * obs_dim)
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
        self.time_step = 0
        self.step_delta_v = 0.0
        self.vel_violation_reward_sum = 0.0
        return self._get_obs(), info

    def velocity_limit(self, distance: float) -> float:
        """Distance-scaled safe speed limit (SafeRL DockingVelocityLimit:
        0.2 + slope * n * dist). Shared by the speed penalty, the start-state
        rejection, and the optional SafeRL observation.
        """
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
        """Return the observation: the 7-element state, plus [speed,
        max_vel_limit] when saferl_obs is on, and the braking margin
        (distance minus stopping distance) when saferl_braking_margin_obs
        is on.

        The timestep is always normalized by max_episode_len. With
        normalize_obs on, positions and velocities are also divided by
        SafeRL's static constants so every input lands near [-1, 1];
        raw values (positions up to 150, timesteps up to 5,000) are out
        of distribution for weights trained on earlier/smaller stages
        and can destabilize training as the curriculum grows.
        """
        obs = np.copy(self.state)
        obs[6] = obs[6] / self.max_episode_len
        if self.normalize_obs:
            obs[0:3] = obs[0:3] / OBS_POS_NORM
            obs[3:6] = obs[3:6] / OBS_VEL_NORM

        if not self.saferl_obs and not self.saferl_braking_margin_obs:
            return obs

        distance = vec_norm(self.state[0:3])
        speed = vec_norm(self.state[3:6])
        if self.saferl_obs:
            # These two SafeRL features use a divisor of 1, so they stay unscaled
            # even when normalize_obs is enabled.
            max_vel_limit = self.velocity_limit(distance)
            obs = np.concatenate([obs, np.array([speed, max_vel_limit])])
        if self.saferl_braking_margin_obs:
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
        # Sample directly from the env's own seeded generator instead of
        # Box.sample(), since Box has no public way to reuse an external
        # generator in this Gymnasium version.
        sampled_state = self.np_random.uniform(low=low, high=high)
        rel_dist = vec_norm(sampled_state[0:3])
        rel_speed = vec_norm(sampled_state[3:6])
        # Reject if too close, too far, or moving too fast for a safe start.
        if (rel_dist < self.min_start_dist or
                rel_dist > self.max_start_dist or
                rel_speed > self.velocity_limit(rel_dist)):
            return self.sample_state_space()  # TODO: recursion error if bounds are very tight
        # Append timestep counter as the 7th state element.
        return np.concatenate([sampled_state, np.array([0])])

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
        self.fuel_used += vec_norm(action) / self.m * self.step_len
        # SafeRL-style per-step delta-V (L1 thrust / mass * step).
        self.step_delta_v = float(np.sum(np.abs(action))) / self.m * self.step_len
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

        Speed over the limit is handled two ways. The soft proximity-scaled
        penalty always applies. With saferl_vel_constraint on, each
        violating step also pays a graded penalty into a per-episode sum,
        and the episode fails once the sum reaches saferl_vel_budget
        (SafeRL semantics: no instant termination, and no extra -1 when
        the budget runs out since the penalties already total -5).

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

        # Budgeted velocity constraint: accumulate the graded penalty
        # while violating, regardless of reward structure, so the budget
        # termination behaves the same in dense and sparse modes.
        violation_penalty = 0.0
        if self.saferl_vel_constraint and current_speed > speed_limit:
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

        if docked:
            print("WIN!")

        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0

        if self.reward_structure == "dense":
            if docked:
                tot_step_rew += 1
                if self.saferl_success_time_bonus:
                    # SafeRL success time bonus: up to +1 extra for
                    # docking with time to spare.
                    tot_step_rew += 1 - self.state[6] / self.max_episode_len
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
                # Positive for closing, negative for opening. dist_coeff is
                # negative, so the delta must be (current - prev), matching
                # drift_env.py. Reversing it inverts the sign and trains the
                # agent to back away / rush the target.
                approach_reward = self.dist_coeff * (current_distance - prev_distance)
            # Speed penalty scaled by proximity so the agent can accelerate
            # freely at long range and only needs precise speed control
            # close to the target. Braking zone sized by actual stopping
            # distance at the current speed, not a fixed 10m, so going
            # fast starts the penalty from farther out.
            braking_zone = max(self.dock_dist, self.stopping_distance(current_speed))
            # Floor added so velocity penalty isn't near-zero at range.
            # Previously this let the agent build speed until close.
            approach_factor = max(0.3, min(1.0, braking_zone / max(current_distance, 1.0)))
            vel_penalty = self.vel_penalty_coeff * max(current_speed - speed_limit, 0) * approach_factor
            tot_step_rew += vel_penalty + approach_reward + self.time_penalty + violation_penalty
            if self.saferl_delta_v_penalty:
                # SafeRL delta-V fuel penalty (scale * step delta-V).
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
        """Check if the deputy has crashed into the chief.

        Returns:
            True if within docking distance but moving too fast to dock safely.
        """
        return (vec_norm(self.state[0:3]) < self.dock_dist and
                vec_norm(self.state[3:6]) >= self.dock_speed)

    def is_unsafe(self) -> bool:
        """True if current speed is above the distance-based limit from 
        velocity_limit().

        When saferl_vel_constraint is on, violating steps also add to
        vel_violation_reward_sum in rewards(); otherwise only the soft
        proximity-scaled penalty applies.
        """
        current_distance = float(vec_norm(self.state[0:3]))
        current_speed = float(vec_norm(self.state[3:6]))
        return current_speed > self.velocity_limit(current_distance)

    def vel_budget_exhausted(self) -> bool:
        """True once accumulated velocity-violation penalties reach the
        budget. The terminal failure condition of the budgeted constraint.
        """
        return (self.saferl_vel_constraint and
                self.vel_violation_reward_sum <= self.saferl_vel_budget)

    def is_out_of_fuel(self) -> bool:
        """Check if the deputy has exceeded its fuel budget.

        Returns:
            True if cumulative fuel used exceeds max_dv.
        """
        return self.fuel_used > self.max_dv

    def is_out_of_bounds(self) -> bool:
        """Check if the deputy has left the allowed spatial region.

        Uses sphere boundary (vec_norm) consistent with drift_env.py,
        rather than the original box boundary (np.max(np.abs(...))).

        Returns:
            True if distance from origin exceeds max_boundary_box.
        """
        return vec_norm(self.state[0:3]) > self.max_boundary_box

    def is_out_of_time(self) -> bool:
        """Check if the episode has reached the step limit.

        Reads the timestep counter from state[6] to stay consistent
        with how propagate() tracks time.

        Returns:
            True if the step count has reached max_episode_len.
        """
        return self.state[6] >= self.max_episode_len  # TODO: > or >= ?

    def only_oot(self) -> bool:
        """Check if out-of-time is the only active termination condition.

        Used by DriftTrainEnv and DriftTestEnv to decide whether to
        still check for a drift opportunity when time runs out.

        Returns:
            True if the episode is only ending due to time.
        """
        return (not (self.is_docked() or self.is_crashed() or
                     self.is_out_of_bounds() or self.is_out_of_fuel())) \
               and self.is_out_of_time()

    def term_other_than_oot(self) -> bool:
        """Check if any termination condition other than time is active.

        Used by drift lookahead logic to detect when an episode has
        ended for a reason other than running out of time.

        Returns:
            True if the episode has ended for a reason other than time.
        """
        return (self.is_docked() or self.is_crashed() or
                self.is_out_of_bounds() or self.is_out_of_fuel())

    def close(self) -> None:
        """Clean up environment resources. Required by Gymnasium interface."""
        pass
