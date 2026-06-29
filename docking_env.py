import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import norm as vec_norm


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
        time_penalty (float): Reward penalty applied every step to encourage speed.
        dist_coeff (float): Reward coefficient for change in distance each step.
        vel_penalty_coeff (float): Reward coefficient for exceeding the speed limit.
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
                 time_step=None):
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

        self.n = 0.001027          # mean motion of chief orbit (rad/s)
        self.m = 12                # deputy mass (kg)
        self.time_penalty = -0.005  # TODO originally 0.0005
        self.dist_coeff = -0.005  # TODO originally 0.0005
        self.vel_penalty_coeff = -0.0075

        self.action_space = spaces.Box(
            low=np.array([-self.u_max] * 3),
            high=np.array([self.u_max] * 3)
        )
        # Observation is unbounded; the environment handles out-of-bounds
        # termination internally rather than clipping the observation.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 7),
            high=np.array([np.inf] * 7)
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
        np.random.seed(seed)
        info = {}
        if self.fixed_start:
            self.state = np.copy(self.fixed_state)
        else:
            self.state = self.sample_state_space()
        self.fuel_used = 0
        self.time_step = 0
        return self.state, info

    def sample_state_space(self) -> np.ndarray:
        """Sample a valid random starting state.

        Samples position and velocity uniformly, then rejects and
        resamples if the position is out of the allowed range or the
        initial speed exceeds the safety speed limit.

        Returns:
            A valid 7-element starting state with timestep counter set to 0.
        """
        initial_state_space = spaces.Box(
            low=np.array([-self.max_start_dist] * 3 + [-self.max_start_speed] * 3),
            high=np.array([self.max_start_dist] * 3 + [self.max_start_speed] * 3)
        )
        sampled_state = initial_state_space.sample()
        rel_dist = vec_norm(sampled_state[0:3])
        rel_speed = vec_norm(sampled_state[3:6])
        # Reject if too close, too far, or moving too fast for a safe start.
        if (rel_dist < self.min_start_dist or
                rel_dist > self.max_start_dist or
                rel_speed > 0.2 + 2 * self.n * rel_dist):
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
        reward, terminated, truncated = self.rewards(old_state)
        info = {}
        return self.state, reward, terminated, truncated, info

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

        Unlike drift_env.py, this environment penalizes unsafe speed
        instead of terminating on it. This keeps episodes alive longer
        for a direct-docking agent that has not yet learned speed control.

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

        # Out-of-time is truncation, not termination, to preserve the
        # episode value estimate in PPO.
        term = docked or crashed or out_of_bounds or out_of_fuel
        trunc = False if term else out_of_time

        if docked:
            print("WIN!")

        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0

        if self.reward_structure == "dense":
            if docked:
                tot_step_rew += 1
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time:
                tot_step_rew += -1

            current_distance = float(vec_norm(self.state[0:3]))
            prev_distance = float(vec_norm(last_state[0:3]))

            # Penalize moving away from the chief.
            prox_penalty = self.dist_coeff * (current_distance - prev_distance)

            # Penalize exceeding the speed limit; zero penalty if within limit.
            current_speed = float(vec_norm(self.state[3:6]))
            speed_limit = 0.2 + (2 * self.n) * current_distance
            vel_penalty = self.vel_penalty_coeff * max(current_speed - speed_limit, 0)

            tot_step_rew += vel_penalty + prox_penalty + self.time_penalty

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
