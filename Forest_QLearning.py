import numpy as np
import matplotlib.pyplot as plt
from hiive.mdptoolbox import mdp, example
import seaborn as sns
import time

def initialize_forest(size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r1 = 10
    r2 = 2
    p = 0.1

    P, R = example.forest(S=size, r1=r1, r2=r2, p=p)
    return P, R

def plot_policy(policy, size, title, filename):
    fig, ax = plt.subplots(figsize=(8, 12))

    cmap = sns.color_palette("viridis", as_cmap=True)

    unique_actions = np.unique(policy)
    sns.heatmap(np.array(policy).reshape(-1, 1), annot=True, cmap=cmap, cbar=True,
                linewidths=0.5, ax=ax, yticklabels=range(size),
                linecolor='grey')

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.5 + x for x in range(len(unique_actions))])
    colorbar.set_ticklabels(['Action ' + str(action) for action in unique_actions])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Action', fontsize=14)
    ax.set_ylabel('State', fontsize=14)

    plt.savefig(f'Images/{filename}', dpi=300)
    plt.close()

def plot_value_function(values, title, filename):
    num_states = len(values)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_states + 1), values, linestyle='-')
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.xticks(range(1, num_states + 1, max(1, num_states // 10)))
    plt.grid(visible=True, which='major', axis='y')
    plt.savefig(f'Images/{filename}')
    plt.close()

def plot_reward_per_iteration(rewards, title, filename):
    window_size = 50
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, linestyle='-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Smoothed Total Reward')
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def plot_q_value_convergence(q_values, title, filename):
    window_size = 50
    smoothed_values = np.convolve(q_values, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_values, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Smoothed Max Q-Value')
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def get_stats(data, name="Data"):
    print(f"\n{name} Statistics:")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}, Median: {np.median(data)}, Std: {np.std(data)}")

def perform_q_learning(P, R, discount_factor=0.99, n_iter=10000):
    start_time = time.time()
    ql = mdp.QLearning(P, R, discount_factor, n_iter=n_iter)
    ql.run()
    end_time = time.time()
    print(f"Q-Learning Runtime: {end_time - start_time} seconds")

    max_q_values = [np.max(ql.Q[i]) for i in range(len(ql.Q))]
    rewards = [stat['Reward'] for stat in ql.run_stats]

    get_stats(max_q_values, "Max Q-Values")
    get_stats(rewards, "Rewards")

    plot_title = f"Value Function with Discount Factor {discount_factor}"
    filename = f"Forest_QL_Value_Function_DF_{discount_factor}.png"
    plot_value_function(np.max(ql.Q, axis=1), plot_title, filename)

    plot_title = f"Reward Per Iteration with Discount Factor {discount_factor}"
    filename = f"Forest_QL_Reward_Iter_DF_{discount_factor}.png"
    plot_reward_per_iteration(rewards, plot_title, filename)

    plot_title = f"Q-Value Convergence with Discount Factor {discount_factor}"
    filename = f"Forest_QL_Convergence_DF_{discount_factor}.png"
    plot_q_value_convergence(max_q_values, plot_title, filename)

    if np.allclose(max_q_values[-10:], max_q_values[-1], atol=0.01):
        print("Convergence achieved in Q-values.")
    else:
        print("Convergence not achieved in Q-values.")
    return np.max(ql.Q, axis=1), ql.policy, n_iter

def run_forest_q_learning(size, discount_factors, n_iter=10000, seed=None):
    P, R = initialize_forest(size, seed)

    for discount_factor in discount_factors:
        print(f"\nRunning Q-Learning for size: {size}, discount factor: {discount_factor}")
        values, policy, iterations = perform_q_learning(P, R, discount_factor, n_iter)

        policy_title = f"Policy for Forest Size {size} with Discount Factor {discount_factor}"
        policy_filename = f"Forest_QL_{size}_DF_{discount_factor}.png"
        plot_policy(policy, size, policy_title, policy_filename)

forest_sizes = [500, 1000]
discount_factors = [0.9, 0.99, 0.999]

for size in forest_sizes:
    run_forest_q_learning(size, discount_factors, seed=20)