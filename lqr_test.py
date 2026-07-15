from drift_env import SpaceCraftDockingEnv3D
from initial_trainer import get_curriculum
from old.old_drift_env import SpaceCraftDockingEnv3D
import scipy
import numpy as np


def discrete_lqr(A_c, B_c, Q, R, dt):
    """
    Designs a discrete-time LQR controller for a continuous-time system.

    Parameters:
    A_c (ndarray): Continuous-time system matrix (n x n)
    B_c (ndarray): Continuous-time input matrix (n x m)
    Q   (ndarray): State cost matrix (n x n)
    R   (ndarray): Input cost matrix (m x m)
    dt  (float):   Sampling time step

    Returns:
    K   (ndarray): Discrete-time optimal feedback gain matrix (m x n)
    P   (ndarray): Solution to the Riccati equation (n x n)
    """
    # 1. Discretize the continuous-time system matrices (Zero-Order Hold)
    # Using scipy's built-in Matrix Exponential method for exact discretization
    n = A_c.shape[0]
    m = B_c.shape[1]

    # Construct the block matrix for simultaneous discretization of A and B
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c

    # Exponential of the block matrix scaled by dt
    M_exp = scipy.linalg.expm(M * dt)

    # Extract discrete matrices
    A_d = M_exp[:n, :n]
    B_d = M_exp[:n, n:]

    # 2. Solve the Discrete-Time Algebraic Riccati Equation (DARE)
    # Equation: P = A^T P A - (A^T P B)(R + B^T P B)^-1 (B^T P A) + Q
    P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)

    # 3. Compute the optimal feedback gain matrix K
    # K = (R + B^T P B)^-1 (B^T P A)
    K = scipy.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)

    return K, P


env = SpaceCraftDockingEnv3D()
obs, info = env.reset()
done = False


n = 0.001027
m = 12

A_c = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [(3 * n ** 2), 0, 0, 0, 2 * n, 0],
              [0, 0, 0, -2 * n, 0, 0],
              [0, 0, - (n ** 2), 0, 0, 0]])
B_c = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1 / m, 0, 0],
              [0, 1 / m, 0],
              [0, 0, 1 / m]])


# 1. Define physical intuition scales
max_pos_error = 1  # We want to stay within 0.2 meters of target
max_vel_error = 1  # We tolerate up to 1.0 m/s deviations during movement
max_control   = 0.0001  # Max force/command we want to easily exert

# 2. Construct Q (6x6)
q_pos = 1.0 / (max_pos_error ** 2)  # 25.0
q_vel = 1.0 / (max_vel_error ** 2)  # 1.0

Q = np.diag([q_pos, q_pos, q_pos,  # X, Y, Z Position penalties
             q_vel, q_vel, q_vel]) # X, Y, Z Velocity penalties

# 3. Construct R (Assuming 3 control inputs, e.g., Forces fx, fy, fz)
r_val = 1.0 / (max_control ** 2)    # 0.01
R = np.diag([r_val, r_val, r_val])

dt = 1

K, P = discrete_lqr(A_c, B_c, Q, R, dt)

print(obs)
action_sum = 0
while not done:
    action = -K @ obs[0:6]
    action_sum += np.linalg.norm(action) / m * dt
    obs, reward, term, trunc, info = env.step(action)
    print(np.linalg.norm(action))
    print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]), obs[6], done)
    print(np.linalg.norm(env.state[0:3]), np.linalg.norm(env.state[3:6]))
    print(env.is_docked(),
          env.is_crashed(),
          env.is_unsafe(),
          env.is_out_of_time(),
          env.is_out_of_fuel(),
          env.is_out_of_bounds()
          )
    print()
    done = term or trunc
    print(done)
print(action_sum)

