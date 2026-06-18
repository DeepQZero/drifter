import gymnasium as gym

from drift_env import DriftTrainEnv
import typing as tt

import numpy as np
from stable_baselines3 import PPO

def curriculum_learn(model_id: int):
    epoch = 0
    for curr in range(10):
        print('Starting Curriculum: ', curr)
        configs, threshold = get_curriculum(curr)
        env = DriftTrainEnv(**configs)
        if curr == 0:
            model = PPO('MlpPolicy', env,
                        learning_rate=0.0003,
                        ent_coef=0.01,
                        gamma=1.00,
                        verbose=1)
        else:
            model = PPO.load('data/checkpoints/safe_ppo_model_'+str(
                model_id)+'_'+str(curr-1)+'_'+str(epoch),  env=env)
        score = 0
        epoch = -1
        while score < threshold:
            epoch += 1
            # TODO what happens if model diverges?
            model.learn(total_timesteps=25_000)
            save_path = 'data/checkpoints/safe_ppo_model_'+str(
                model_id)+'_'+str(curr)+'_'+str(epoch)
            model.save(save_path)
            score = test_model(save_path, curr)
            print('Saved Model: ', save_path, ' Score: ', score)

def test_model(path, curriculum) -> float:
    configs, _ = get_curriculum(curriculum)
    env = DriftTrainEnv(**configs)
    model = PPO.load(path, env=env)
    all_rews, all_fuels, all_docks = [], [], []
    for i in range(1_000):
        done = False
        obs, info = env.reset()
        epi_fuel, epi_reward = 0, 0
        is_drift = False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, term, trunc, info = env.step(action)
            epi_reward += reward
            epi_fuel += np.linalg.norm(action)  # TODO
            is_drift, _ = env.det_drift()
            done = term or trunc or is_drift
            if done:
                if env.env.is_docked() or is_drift:
                    all_docks.append(1)
                else:
                    all_docks.append(0)
                all_rews.append(epi_reward)
                all_fuels.append(epi_fuel)
    print('Model Stats for Curriculum:', curriculum)
    print('Dock: ', np.mean(all_docks),
          'Reward: ', np.mean(all_rews),
          'Fuel: ', np.mean(all_fuels), np.median(all_fuels)
         )
    return float(np.mean(all_docks))


def get_curriculum(curriculum: int) -> tt.Tuple[dict, float]:
    configs = {
        'fixed_start' : False,
        'fixed_state' : np.array([100, 0, 0, 0, 0, 0, 0]),  # TODO change
        'reward_structure' : "dense",
        'max_episode_len' : 10_000,
        'max_lookahead_len' : 1_000,
        'max_boundary_box' : 200.0,
        'max_control' : 1.0,
        'max_total_dv' : 1_000.0,
        'pos_thresh' : 0.5,
        'speed_thresh' : 0.2,
        'min_init_pos_bound' : 100.0,
        'max_init_pos_bound' : 150.0,
        'max_init_vel_bound' : 0.5,
        'step_len' : 1,
        'fuel_used' : None,
        'time_step' : None,
        'drift_step_len' : 1
    }
    threshold = 0.99
    if curriculum >= 0:
        threshold = 0.95
        configs['pos_thresh'] = 0.5
        configs['speed_thresh'] = 0.2
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 20
        configs['max_boundary_box'] = 3
        configs['min_init_pos_bound'] = 0.5
        configs['max_init_pos_bound'] = 1
        configs['max_init_vel_bound'] = 0.2
    if curriculum >= 1:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 10
        configs['min_init_pos_bound'] = 0.5
        configs['max_init_pos_bound'] = 2.5
        configs['max_init_vel_bound'] = 0.2
    if curriculum >= 2:
        threshold = 0.95
        configs['pos_thresh'] = 2.5
        configs['speed_thresh'] = 0.2
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 20
        configs['min_init_pos_bound'] = 2.5
        configs['max_init_pos_bound'] = 5
        configs['max_init_vel_bound'] = 0.2
    if curriculum >= 3:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 100
        configs['max_boundary_box'] = 30
        configs['min_init_pos_bound'] = 2.5
        configs['max_init_pos_bound'] = 10
        configs['max_init_vel_bound'] = 0.2
    if curriculum >= 4:
        threshold = 0.95
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.22
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 10
        configs['max_boundary_box'] = 50
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 20
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10
    if curriculum >= 5:
        threshold = 0.95
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 25
        configs['max_boundary_box'] = 40
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 35
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10
    if curriculum >= 6:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 60
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 50
        configs['max_init_vel_bound'] = 0.3
        configs['drift_step_len'] = 10
    if curriculum >= 7:
        threshold = 0.95
        configs['pos_thresh'] = 50
        configs['speed_thresh'] = 0.3
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 110
        configs['min_init_pos_bound'] = 50
        configs['max_init_pos_bound'] = 100
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 10
    if curriculum >= 8:
        threshold = 0.95
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 100
        configs['max_boundary_box'] = 200
        configs['min_init_pos_bound'] = 50
        configs['max_init_pos_bound'] = 150
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 10
    if curriculum >= 9:
        threshold = 0.985
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 100
        configs['max_boundary_box'] = 200
        configs['min_init_pos_bound'] = 100
        configs['max_init_pos_bound'] = 150
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 10
    if curriculum >= 10:  # Testing Curriculum
        configs['pos_thresh'] = 50
        configs['speed_thresh'] = 0.3
        configs['max_episode_len'] = 9_000  # TODO think about changing
        configs['max_lookahead_len'] = 1000
        configs['max_boundary_box'] = 200
        configs['min_init_pos_bound'] = 100
        configs['max_init_pos_bound'] = 150
        configs['max_init_vel_bound'] = 0.5
        configs['drift_step_len'] = 1
    return configs, threshold

if __name__ == "__main__":
    # model_id = 5
    # curriculum_learn(model_id)
    test_model('data/checkpoints/safe_ppo_model_5_9_2.zip', 9)