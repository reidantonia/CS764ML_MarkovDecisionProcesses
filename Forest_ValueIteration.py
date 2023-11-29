import numpy as np
import matplotlib.pyplot as plt
from hiive.mdptoolbox import mdp, example
import seaborn as sns
import time

def initialize_forest(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r1, r2, p = 10, 2, 0.1
    P, R = example.forest(S=size, r1=r1, r2=r2, p=p)
    return P, R

def plot_policy(policy, size, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(np.array(policy).reshape(-1, 1), annot=False, cmap='coolwarm', cbar=False, linewidths=0, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Action', fontsize=10)
    ax.set_ylabel('State', fontsize=10)
    ax.set_yticks(np.arange(0, size, max(1, size // 10)))
    ax.set_yticklabels(np.arange(0, size, max(1, size // 10)).astype(int), fontsize=8)
    ax.set_xticks([])
    fig.tight_layout()
    plt.savefig(f'Images/{filename}', dpi=300)
    plt.close()

def plot_value_function(values, title, filename):
    num_states = len(values)
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, num_states + 1), values, marker='o')
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Value')
    if num_states <= 50:
        plt.xticks(range(1, num_states + 1))
    else:
        plt.xticks(range(1, num_states + 1, num_states // 10))
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def plot_convergence(run_stats, title, filename):
    iterations = [stat['Iteration'] for stat in run_stats]
    max_v = [stat['Max V'] for stat in run_stats]
    plt.figure(figsize=(6, 6))
    plt.plot(iterations, max_v, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Max Value')
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def plot_rewards(run_stats, title, filename):
    iterations = [stat['Iteration'] for stat in run_stats]
    rewards = [stat['Reward'] for stat in run_stats]
    plt.figure(figsize=(6, 6))
    plt.plot(iterations, rewards, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def calculate_averages(vi):
    average_reward = np.mean([s['Reward'] for s in vi.run_stats if 'Reward' in s])
    average_value = np.mean(vi.V)
    return average_reward, average_value

def print_summary(size, discount_factor, iterations, vi):
    average_reward = np.mean([s['Reward'] for s in vi.run_stats if 'Reward' in s])
    max_reward = np.max([s['Reward'] for s in vi.run_stats if 'Reward' in s])
    max_value = np.max(vi.V)
    mean_value = np.mean(vi.V)
    print(f"\nModel: Forest Management")
    print(f"Size: {size}, Discount Factor: {discount_factor}")
    print(f"Iterations to Converge: {iterations}")
    print(f"Average Reward: {average_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Max Value: {max_value}")
    print(f"Mean Value: {mean_value}")

def plot_value_distribution(values, title, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'Images/{filename}')
    plt.close()

def perform_value_iteration(P, R, discount_factor=0.99):
    start_time = time.time()  # Start timing
    vi = mdp.ValueIteration(P, R, discount_factor)
    vi.run()
    end_time = time.time()  # End timing
    convergence_time = end_time - start_time  # Calculate time to convergence
    print(f"Time to convergence: {convergence_time:.2f} seconds")


    for i, stat in enumerate(vi.run_stats):
        print(f"Iteration {i+1}: {stat}")
    plot_title = f"Value Function with Discount Factor {discount_factor}"
    filename = f"Forest_VI_Value_Function_DF_{discount_factor}.png"
    plot_value_function(vi.V, plot_title, filename)
    return vi.V, vi.policy, vi.iter

def run_forest_value_iteration(size, discount_factors, seed=None):
    P, R = initialize_forest(size, seed)
    for discount_factor in discount_factors:
        start_time = time.time()  # Start timing
        vi = mdp.ValueIteration(P, R, discount_factor)
        vi.run()
        end_time = time.time()  # End timing
        convergence_time = end_time - start_time  # Calculate time to convergence
        print(f"Size: {size}, Discount Factor: {discount_factor}, Time to Convergence: {convergence_time:.2f} seconds")



        values, policy, iterations = vi.V, vi.policy, len(vi.run_stats)
        plot_title = f"Value Function with Discount Factor {discount_factor}"
        value_filename = f"Forest_VI_Value_Function_DF_{discount_factor}.png"
        plot_value_function(values, plot_title, value_filename)
        policy_title = f"Policy for Forest Size {size} with Discount Factor {discount_factor}"
        policy_filename = f"Forest_VI_{size}_Policy_DF_{discount_factor}.png"
        plot_policy(policy, size, policy_title, policy_filename)
        convergence_title = f"Convergence with Discount Factor {discount_factor}"
        convergence_filename = f"Convergence_DF_{discount_factor}.png"
        plot_convergence(vi.run_stats, convergence_title, convergence_filename)
        rewards_title = f"Rewards with Discount Factor {discount_factor}"
        rewards_filename = f"Rewards_DF_{discount_factor}.png"
        plot_rewards(vi.run_stats, rewards_title, rewards_filename)
        distribution_title = f"Value Distribution with Discount Factor {discount_factor}"
        distribution_filename = f"Value_Distribution_DF_{discount_factor}.png"
        plot_value_distribution(values, distribution_title, distribution_filename)
        print_summary(size, discount_factor, iterations, vi)
        print(f"Size: {size}, Discount Factor: {discount_factor}, Iterations: {iterations}")

forest_sizes = [500, 1000]
discount_factors = [0.9, 0.99, 0.999]

for size in forest_sizes:
    run_forest_value_iteration(size, discount_factors, seed=20)