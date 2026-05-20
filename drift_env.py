import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import norm as vec_norm


class SpaceCraftDockingEnv3D(gym.Env):
    def __init__(self,
                 fixed_start=False,
                 reward_structure="dense",
                 max_episode_len=5,
                 max_lookahead_len=300,
                 max_boundary_box=175,
                 max_control=1,
                 max_total_dv=2_500,
                 pos_thresh=10.0,
                 speed_thresh=1.0,
                 min_init_pos_bound=10,
                 max_init_pos_bound=150,
                 max_init_vel_bound=0.1,
                 fixed_state=np.array([100, 0, 0, 0, 0, 0]),
                 step_len = 1,
                 fuel_used=None,
                 time_step=None,
                 drift_step_len=1
                 ):
        self.lookahead_len = max_lookahead_len
        self.fixed_start = fixed_start
        self.fixed_state = fixed_state
        self.abs_min_init_dist = min_init_pos_bound
        self.abs_max_init_dist = max_init_pos_bound
        self.abs_max_vel = max_init_vel_bound
        self.n = 0.001027
        self.m = 12
        self.step_len = step_len
        self.max_episode_len = max_episode_len
        self.max_boundary_box = max_boundary_box
        self.max_total_dv = max_total_dv
        self.docking_pos_thresh = pos_thresh
        self.docking_speed_thresh = speed_thresh
        self.u_max = max_control
        self.reward_structure = reward_structure
        self.time_penalty = -0.0005
        self.proximity_penalty_coeff = -0.0005
        self.min_vel_penalty_coeff = -0.0075
        self.action_space = spaces.Box(
            low=np.array([-self.u_max]*3),
            high=np.array([self.u_max]*3)
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*6),
            high=np.array([np.inf]*6)
        )
        self.state = None
        self.fuel_used = fuel_used
        self.time_step = time_step
        self.drift_step_len = drift_step_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        info = {}
        if self.fixed_start:
            self.state = np.copy(self.fixed_state)
        else:
            self.state = self.sample_state_space()
        self.fuel_used = 0
        self.time_step = 0
        # self.lookahead_len = int(2*np.linalg.norm(self.state[0:3]))+5 # TODO
        # self.max_boundary_box = int(np.linalg.norm(self.state[0:3])*1.2)+2 #
        # TODO
        return self.state, info

    def sample_state_space(self):
        initial_state_space = spaces.Box(low=np.array([-self.abs_max_init_dist, -self.abs_max_init_dist,
                                                       -self.abs_max_init_dist, -self.abs_max_vel,
                                                       -self.abs_max_vel, -self.abs_max_vel]),
                                        high=np.array([self.abs_max_init_dist, self.abs_max_init_dist,
                                                       self.abs_max_init_dist, self.abs_max_vel,
                                                       self.abs_max_vel, self.abs_max_vel]))
        sampled_state = initial_state_space.sample()
        if not (self.abs_min_init_dist <= vec_norm(sampled_state) <= self.abs_max_init_dist):
            return self.sample_state_space()
        return sampled_state

    def step(self, action, drift=False):
        old_state = np.copy(self.state)
        self.time_step += 1
        self.state = self.propagate(action, drift)
        self.fuel_used += vec_norm(action)
        reward, terminated, truncated = self.rewards(old_state)
        info = {}
        return self.state, reward, terminated, truncated, info

    def rewards(self, last_state):
        tot_step_rew = 0
        docked = self.is_docked()
        crashed = self.is_crashed()
        out_of_time = self.is_out_of_time()
        out_of_fuel = self.is_out_of_fuel()
        out_of_bounds = self.is_out_of_bounds()
        term = docked or crashed or out_of_bounds or out_of_fuel
        trunc = False if term else out_of_time
        # if docked:
        #     print("WIN!")
        if self.reward_structure == "sparse":
            tot_step_rew = 1 if docked else 0

        if self.reward_structure == "dense":
            if docked:
                tot_step_rew += 1
            elif crashed:
                tot_step_rew += -1
            elif out_of_bounds or out_of_fuel or out_of_time:
                tot_step_rew += -1

            current_distance = vec_norm(self.state[0:3])
            prev_distance = vec_norm(last_state[0:3])

            current_speed = vec_norm(self.state[3:6])
            speed_limit = 0.2 + (2 * self.n) * current_distance
            vel_penalty = (self.min_vel_penalty_coeff *
                           max(current_speed - speed_limit, 0))

            prox_penalty = (self.proximity_penalty_coeff *
                            (current_distance - prev_distance))

            tot_step_rew += (prox_penalty + self.time_penalty)

            term = term or (max(current_speed - speed_limit, 0) > 0)  # TODO put in sparse
            if max(current_speed - speed_limit, 0) > 0:
                print("UNSAFE!")

        return tot_step_rew, term, trunc

    def close(self):
        pass

    def is_docked(self):
        return (vec_norm(self.state[0:3]) < self.docking_pos_thresh and
                vec_norm(self.state[3:6]) < self.docking_speed_thresh)

    def is_crashed(self):
        return (vec_norm(self.state[0:3]) < self.docking_pos_thresh) and \
            (vec_norm(self.state[3:6]) >= self.docking_speed_thresh)

    def is_out_of_fuel(self):
        return self.fuel_used > self.max_total_dv

    def is_out_of_bounds(self):
        return np.max(np.abs(self.state[0:3])) > self.max_boundary_box

    def is_out_of_time(self):
        return self.time_step >= self.max_episode_len

    def propagate(self, action, drift=False):
        step_len = self.step_len if not drift else self.drift_step_len
        t_span = (self.time_step, self.time_step + step_len)
        result = solve_ivp(self.dynamics, t_span, self.state,
                           args=(action,), method='RK45')
        time_points, state_vectors = result.t, result.y
        new_state = state_vectors[:, -1]
        return new_state

    def dynamics(self, t, x, u):

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


class DriftEnv(gym.Env):
    def __init__(self, **kwargs):
        self.env = SpaceCraftDockingEnv3D(
            **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, act):
        obs, rew, term, trunc, info = self.env.step(act)
        if self.env.is_docked():
            print('WIN!')
        if not term or trunc:
            is_drift, the_time = self.det_drift()
            if is_drift:
                print('DRIFTED! ', the_time)
                rew += 10
                term = True
        return obs, rew, term, trunc, info

    def det_drift(self):
        new_env = copy.deepcopy(self.env)
        t = 0
        for j in range(self.env.lookahead_len):  # TODO create hyperparameter
            drift_action = np.array([0.0, 0.0, 0.0])
            t += 1
            obs, rew, term, trunc, info = new_env.step(drift_action, True)
            if term:
                return new_env.is_docked(), t
            elif new_env.time_step >= new_env.lookahead_len:
                return False, t
        return False, t


class DriftEnv2(gym.Env):
    def __init__(self, **kwargs):
        self.env = SpaceCraftDockingEnv3D(
            **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.is_drifting = False

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, act):
        obs, rew, term, trunc, info = self.env.step(act)
        if self.env.is_docked():
            print('WIN!')
        if not term or trunc:
            if not self.is_drifting:
                is_drift, the_time = self.det_drift()
                if is_drift:
                    self.is_drifting = True
        return obs, rew, term, trunc, info

    def det_drift(self):
        new_env = copy.deepcopy(self.env)
        t = 0
        for j in range(self.env.lookahead_len):  # TODO create hyperparameter
            drift_action = np.array([0.0, 0.0, 0.0])
            t += 1
            obs, rew, term, trunc, info = new_env.step(drift_action, True)
            if term:
                return new_env.is_docked(), t
            elif new_env.time_step >= new_env.lookahead_len:
                return False, t
        return False, t