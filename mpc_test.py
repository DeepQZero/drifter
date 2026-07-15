from old.old_drift_env import SpaceCraftDockingEnv3D
import scipy
import numpy as np
import copy
import itertools
from multiprocessing import Pool


def get_lookahead_score(candidate):
    env, actions = candidate
    score = 0
    for act in actions:
        new_act = np.array(act)
        obs, rew, term, trunc, info = env.step(new_act)
        score += rew
        if term or trunc:
            return score
    return score


def det_best_action(env, lookahead_len):
    possibles = list(itertools.product([-1.0, 0.0, 1.0], repeat=3))
    act_seqs = list(itertools.product(possibles, repeat=lookahead_len))
    candidates = [(copy.deepcopy(env), p) for p in act_seqs]
    with Pool(processes=12) as pool:
        results = pool.map(get_lookahead_score, candidates)
    unordered = list(zip(results, act_seqs))
    # print(unordered)
    best_ordered = sorted(unordered, reverse=True)
    return best_ordered[0][1][0]


def main(lookahead_len=5):
    env = SpaceCraftDockingEnv3D()
    done = False
    obs, info = env.reset()
    total_reward = 0
    total_fuel = 0
    while not done:
        action = det_best_action(env, lookahead_len)
        total_fuel += np.linalg.norm(action) / 12 * 1
        obs, rew, term, trunc, info = env.step(action)
        print(np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6]))
        total_reward += rew
        done = term or trunc
    print(total_fuel, total_reward)


if __name__ == "__main__":
    main(2)

