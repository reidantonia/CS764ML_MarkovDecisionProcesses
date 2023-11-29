import numpy as np
import matplotlib.pyplot as plt
from hiive.mdptoolbox import mdp, example
import seaborn as sns

def initialize_forest(size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r1 = 10
    r2 = 2
    p = 0.1
    print(f"Initializing forest with size: {size}, reward for cutting: {r1}, reward for waiting: {r2}, fire probability: {p}")

    P, R = example.forest(S=size, r1=r1, r2=r2, p=p)
    return P, R

def plot_policy(policy, size, title, filename):

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.heatmap(np.array(policy).reshape(-1, 1), annot=False, cmap='Set1', cbar=False, linewidths=0, ax=ax)

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
    plt.savefig(f'Images/function_{filename}')
    plt.close()

def perform_policy_iteration(P, R, discount_factor=0.99):
    pi = mdp.PolicyIteration(P, R, discount_factor)
    pi.run()

    plot_title = f"Value Function with Discount Factor {discount_factor}"
    filename = f"Forest_PI_Value_Function_DF_{discount_factor}.png"
    plot_value_function(pi.V, plot_title, filename)

    return pi.V, pi.policy, pi.iter

def run_forest_policy_iteration(size, discount_factors, seed=None):
    iteration_counts = []
    max_values = []
    mean_values = []

    P, R = initialize_forest(size, seed)

    for discount_factor in discount_factors:
        values, policy, iterations = perform_policy_iteration(P, R, discount_factor)
        policy_title = f"Policy for Forest Size {size} with Discount Factor {discount_factor}"
        policy_filename = f"Forest_PI_{size}_Policy_DF_{discount_factor}.png"
        plot_policy(policy, size, policy_title, policy_filename)

        iteration_counts.append(iterations)
        max_values.append(np.max(values))
        mean_values.append(np.mean(values))

    avg_iterations = np.mean(iteration_counts)
    std_dev_iterations = np.std(iteration_counts)
    avg_max_v = np.mean(max_values)
    avg_mean_v = np.mean(mean_values)
    print(f"Grid Size: {size}, Discount Factor: {discount_factor}")
    print(f"Average Iterations: {avg_iterations}, Std Dev of Iterations: {std_dev_iterations}")
    print(f"Average Max V: {avg_max_v}, Average Mean V: {avg_mean_v}")

forest_sizes = [500, 1000]
discount_factors = [0.9, 0.99, 0.999]

for size in forest_sizes:
    run_forest_policy_iteration(size, discount_factors, seed=20)