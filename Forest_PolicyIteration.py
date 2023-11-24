import numpy as np
import matplotlib.pyplot as plt
import random
from hiive.mdptoolbox import mdp, example
import seaborn

RANDOM_SEED = 20
NUM_RUNS = 100

def initialize_forest(size, seed=RANDOM_SEED):
    random.seed(seed)
    P, R = example.forest(S=size, r1=4, r2=2, p=0.1)
    return P, R
def create_forest_heatmap(values, size, title, filename_suffix, run):
    fig, ax = plt.subplots()

    ax.bar(range(size), values, color='blue')
    ax.set_ylabel('State Values')
    ax.set_xlabel('States')
    ax.set_title(title)

    plt.savefig(f'Images/Forest_{size}_Values{run}_{filename_suffix}.png')
    plt.close()

def perform_forest_policy_iteration(P, R, size, discount_factor=0.99, epsilon=0.01):
    vi = mdp.ValueIteration(P, R, discount_factor, epsilon=epsilon)
    vi.run()

    policy_history = [vi.policy]
    value_history = [vi.V]

    return vi.V, vi.policy, policy_history, value_history

def run_forest_simulations():
    convergence_info = {}
    for size in [210, 300]:
        cumulative_values = []
        cumulative_policies = []
        iteration_counts = []

        P, R = initialize_forest(size)

        for run in range(NUM_RUNS):
            values, policy, policy_history, value_history = perform_forest_policy_iteration(P, R, size)

            cumulative_values.append(values)
            cumulative_policies.append(policy)
            iteration_counts.append(len(policy_history))

            create_forest_heatmap(values, size, f'State Values for {size}x{size} Grid', 'policy_iteration', run)

        avg_iterations = np.mean(iteration_counts)
        convergence_info[size] = {
            'avg_iterations': avg_iterations,
            'values': np.mean(cumulative_values, axis=0),
            'policies': cumulative_policies
        }

        print(f"Size {size} - Average iterations to converge: {avg_iterations}")
        print(f"Size {size} - Average Max V: {np.max(convergence_info[size]['values'])}, Average Mean V: {np.mean(convergence_info[size]['values'])}")

    if all(size in convergence_info for size in [210, 300]):
        print("\nConvergence Comparison:")
        faster_convergence = 210 if convergence_info[210]['avg_iterations'] < convergence_info[300]['avg_iterations'] else 300
        print(f"Grid size {faster_convergence} converges faster on average.")

        same_policy = all(np.array_equal(convergence_info[210]['policies'][i], convergence_info[300]['policies'][i]) for i in range(NUM_RUNS))
        print(f"Do they converge to the same answer? {'Yes' if same_policy else 'No'}")
        print("Larger grid sizes have more states, which generally increases the complexity and iterations needed for convergence.")

run_forest_simulations()
