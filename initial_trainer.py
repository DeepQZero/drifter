import gymnasium as gym

from drift_env import DriftEnv
import typing as tt

import numpy as np
from stable_baselines3 import PPO

def curriculum_learn(model_id: int):
    for curr in range(9, 10):
        print('Starting Curriculum: ', curr)
        configs, train_time = get_curriculum(curr)
        env = DriftEnv(**configs)
        if curr == 0:
            model = PPO('MlpPolicy', env,
                        learning_rate=0.0003,
                        ent_coef=0.01,
                        gamma=1.00,
                        verbose=1)
        else:
            model = PPO.load('data/checkpoints/safe_ppo_model_'+str(
                curr-1)+'_'+str(
                model_id),  env=env)
        model.learn(total_timesteps=train_time)
        save_path = 'data/checkpoints/safe_ppo_model_'+str(curr)+'_'+str(
            model_id)
        model.save(save_path)
        test_model(save_path, curr)
        print('Saved Model: ', save_path)

def test_model(path, curriculum):
    configs, train_time = get_curriculum(curriculum)
    env = DriftEnv(**configs)
    model = PPO.load(path, env=env)
    all_rews, all_fuels, all_docks = [], [], []
    for _ in range(100):
        done = False
        obs, info = env.reset()
        epi_fuel, epi_reward = 0, 0
        while not done:
            action = model.predict(obs)[0]
            obs, reward, term, trunc, info = env.step(action)
            epi_reward += reward
            epi_fuel += np.linalg.norm(action)
            done = term or trunc
            if done:
                if env.env.is_docked():
                    all_docks.append(1)
                else:
                    is_drift, t_t = env.det_drift()
                    if is_drift:
                        all_docks.append(1)
                    else:
                        all_docks.append(0)
                all_rews.append(epi_reward)
                all_fuels.append(epi_fuel)
    print('Model Stats for Curriculum: ', curriculum)
    print('Dock: ', np.mean(all_docks),
          'Reward: ', np.mean(all_rews),
          'Fuel: ', np.mean(all_fuels)
         )
    print()


def get_curriculum(curriculum: int) -> tt.Tuple[dict, int]:
    configs = {
        'fixed_start' : False,
        'reward_structure' : "dense",
        'max_episode_len' : 5,
        'max_lookahead_len' : 10,
        'max_boundary_box' : 3,
        'max_control' : 1,
        'max_total_dv' : 1000,
        'pos_thresh' : 0.5, # TODO was originally at 1
        'speed_thresh' : 0.2,
        'min_init_pos_bound' : 1,
        'max_init_pos_bound' : 2,
        'max_init_vel_bound' : 0.1,
        'fixed_state' : np.array([100, 0, 0, 0, 0, 0]),  # TODO Fix
        'step_len' : 1,
        'fuel_used' : None,
        'time_step' : None,
        'drift_step_len' : 1
    }
    train_time = 50_000
    if curriculum >= 0:
        configs['pos_thresh'] = 1
        configs['speed_thresh'] = 1
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 20
        configs['max_boundary_box'] = 3
        configs['min_init_pos_bound'] = 1
        configs['max_init_pos_bound'] = 2
        configs['max_init_vel_bound'] = 0.1
    if curriculum >= 1:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 30
        configs['max_boundary_box'] = 10
        configs['min_init_pos_bound'] = 1
        configs['max_init_pos_bound'] = 3
        configs['max_init_vel_bound'] = 0.1
    if curriculum >= 2:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 50
        configs['max_boundary_box'] = 10
        configs['min_init_pos_bound'] = 1
        configs['max_init_pos_bound'] = 5
        configs['max_init_vel_bound'] = 0.1
    if curriculum >= 3:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 75
        configs['max_boundary_box'] = 20
        configs['min_init_pos_bound'] = 1
        configs['max_init_pos_bound'] = 7.5
        configs['max_init_vel_bound'] = 0.1
    if curriculum >= 4:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 100
        configs['max_boundary_box'] = 30
        configs['min_init_pos_bound'] = 0.5
        configs['max_init_pos_bound'] = 10
        configs['max_init_vel_bound'] = 0.1
    # if curriculum >= 5:
    #     configs['pos_thresh'] = 0.5
    #     configs['speed_thresh'] = 0.2
    if curriculum >= 5:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 20
        configs['max_boundary_box'] = 50
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 20
        configs['max_init_vel_bound'] = 0.1
        configs['drift_step_len'] = 10
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.2
    if curriculum >= 6:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 20
        configs['max_boundary_box'] = 40
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 35
        configs['max_init_vel_bound'] = 0.1
        configs['drift_step_len'] = 10
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.2
    if curriculum >= 7:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 30
        configs['max_boundary_box'] = 55
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 50
        configs['max_init_vel_bound'] = 0.1
        configs['drift_step_len'] = 10
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.2
    if curriculum >= 8:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 40
        configs['max_boundary_box'] = 80
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 75
        configs['max_init_vel_bound'] = 0.1
        configs['drift_step_len'] = 10
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.2
    if curriculum >= 9:
        configs['max_episode_len'] = 5
        configs['max_lookahead_len'] = 60
        configs['max_boundary_box'] = 110
        configs['min_init_pos_bound'] = 10
        configs['max_init_pos_bound'] = 100
        configs['max_init_vel_bound'] = 0.1
        configs['drift_step_len'] = 10
        configs['pos_thresh'] = 10
        configs['speed_thresh'] = 0.2

    return configs, train_time

if __name__ == "__main__":
    # model_id = 1
    # curriculum_learn(model_id)
    test_model('data/checkpoints/safe_ppo_model_4_1.zip', 4)