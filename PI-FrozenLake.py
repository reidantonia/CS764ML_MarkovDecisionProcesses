import gym
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
MY_SEED = 20

def initialize_environment(size, random_map_seed=MY_SEED):
    if size == 4:
        env = gym.make("FrozenLake-v1")
    else:
        random.seed(random_map_seed)
        random_map = generate_random_map(size=size, p=0.8)
        env = gym.make("FrozenLake-v1", desc=random_map)
    return env.unwrapped

def evaluate_policy(env, V, policy, epsilon=0.0001, gamma=0.99):
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            old_v = V[s]
            V[s] = sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][policy[s]]])
            delta = max(delta, abs(old_v - V[s]))
        if delta < epsilon:
            break

def improve_policy(env, V, policy, gamma=0.99):
    policy_stable = True
    for s in range(env.observation_space.n):
        old_a = policy[s]
        policy[s] = np.argmax([sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]]) for a in range(env.action_space.n)])
        if old_a != policy[s]:
            policy_stable = False
    return policy_stable

def run_policy_iteration(env, gamma=0.99, epsilon=0.0001):
    V = np.zeros(env.observation_space.n)
    policy = np.random.choice(env.action_space.n, env.observation_space.n)
    policy_stable = False
    while not policy_stable:
        evaluate_policy(env, V, policy, epsilon, gamma)
        policy_stable = improve_policy(env, V, policy, gamma)
    return V, policy

def plot_policy(V, policy, size, filename_suffix):
    V_sq = np.reshape(V, (size, size))
    P_sq = np.reshape(policy, (size, size))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(V_sq, cmap='cool')
    fontSize = 20 if size < 10 else 10

    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 2), ha='center', va='center', fontsize=fontSize)
        action = ['LEFT', 'DOWN', 'RIGHT', 'UP'][P_sq[j][i]]
        ax.text(i, j, action, ha='center', va='bottom', fontsize=fontSize)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('State-Value Function and Policy')
    plt.savefig(f'Images/PI-FL-Plot_vals{size}_{filename_suffix}.png')
    plt.show()


def create_heatmap(data, row_labels, col_labels, cbarlabel, title, filename_suffix):
    fig, ax = plt.subplots()
    cbar_kw = {}
    im = ax.imshow(data, cmap="YlGn")
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    plt.title(title)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.savefig(f'Images/FL_PI_{filename_suffix}_{size}.png')
    plt.show()
def run_fl(size):
    """
    Runs the Frozen Lake simulation with Policy Iteration and generates various plots.

    Args:
        size (int): Size of the frozen lake (e.g., 4 for a 4x4 grid).

    Returns:
        None: The function runs the simulation and plots results.
    """
    env = initialize_environment(size)
    V, policy = run_policy_iteration(env)
    plot_policy(V, policy, size, 'Values_Policy')

    # Additional code for generating and saving the heatmaps
    # Assuming the logic for generating gammas, epsilons, and corresponding matrices remains the same
    create_heatmap(per_won_hm, gammas, epsilons, "% Games Won", "Performance Heatmap", "Performance")
    create_heatmap(iters_hm, gammas, epsilons, "# of Iterations", "Iterations Heatmap", "Iterations")
    create_heatmap(time_hm, gammas, epsilons, "Runtime (ms)", "Runtime Heatmap", "Runtime")


run_fl(4)
run_fl(16)
