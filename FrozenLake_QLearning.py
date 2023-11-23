import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map

RANDOM_SEED = 20
NUM_RUNS = 10

def initialize_environment(lake_size, map_seed=RANDOM_SEED):
    random.seed(map_seed)
    lake_map = generate_random_map(size=lake_size, p=0.8) if lake_size != 4 else None
    environment = gym.make("FrozenLake-v1", desc=lake_map)
    return environment.unwrapped

def create_heatmap(values, grid_size, title, filename_suffix, run):

    reshaped_values = np.reshape(values, (grid_size, grid_size))

    fig, ax = plt.subplots()
    cbar_kw = {}

    im = ax.imshow(reshaped_values, cmap="coolwarm", interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel('State Values', rotation=-90, va="bottom")

    for (j, i), val in np.ndenumerate(reshaped_values):
        ax.text(i, j, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black')

    row_labels = col_labels = [str(i) for i in range(grid_size)]
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title(title)

    plt.savefig(f'Images/FrozenLake_QLearning_Values{run}_Size{grid_size}_{filename_suffix}.png')
    plt.close()
def perform_q_learning(environment, discount_factor=0.99, learning_rate=0.1, epsilon=0.1, max_iterations=10000):
    q_table = np.zeros([environment.observation_space.n, environment.action_space.n])
    convergence_info = {'max_values': [], 'mean_values': []}

    for iteration in range(max_iterations):
        initial_state = environment.reset()
        state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = environment.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            step_result = environment.step(action)
            if isinstance(step_result, tuple) and len(step_result) >= 4:
                next_state, reward, done, _ = step_result[:4]
            else:
                raise ValueError("Unexpected format from environment's step method")

            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            state = int(next_state)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state, action] = new_value

        convergence_info['max_values'].append(np.max(q_table))
        convergence_info['mean_values'].append(np.mean(q_table))

    policy = np.argmax(q_table, axis=1)
    return np.max(q_table, axis=1), policy, q_table, convergence_info

def visualize_q_convergence(convergence_info, grid_size, run_number):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(convergence_info['max_values'], label='Max Q')
    plt.xlabel('Iteration')
    plt.ylabel('Max Q-value')
    plt.title(f'Max Q per Iteration for Size-{grid_size} Frozen Lake')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(convergence_info['mean_values'], label='Mean Q')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Q-value')
    plt.title(f'Mean Q per Iteration for Size-{grid_size} Frozen Lake')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Images/FrozenLake_QL_Conv_Size{grid_size}_Run{run_number}.png')
    plt.close()

def run_simulations():
    convergence_info = {}
    for grid_size in [5, 20]:
        cumulative_values = []
        cumulative_policies = []
        iteration_counts = []

        for run in range(NUM_RUNS):
            env = initialize_environment(grid_size)
            values, policy, q_table, q_convergence_info = perform_q_learning(env)

            cumulative_values.append(values)
            cumulative_policies.append(policy)
            iteration_counts.append(len(q_convergence_info['max_values']))

            create_heatmap(values, grid_size, f'State Values for {grid_size}x{grid_size} Grid', 'heatmap', run)
            visualize_q_convergence(q_convergence_info, grid_size, run)

        avg_iterations = np.mean(iteration_counts)
        convergence_info[grid_size] = {
            'avg_iterations': avg_iterations,
            'values': np.mean(cumulative_values, axis=0),
            'policies': cumulative_policies
        }

        print(f"Size {grid_size} - Average iterations to converge: {avg_iterations}")
        print(f"Size {grid_size} - Average Max Q: {np.max(convergence_info[grid_size]['values'])}, Average Mean Q: {np.mean(convergence_info[grid_size]['values'])}")

    if all(size in convergence_info for size in [5, 20]):
        print("\nConvergence Comparison:")
        faster_convergence = 5 if convergence_info[5]['avg_iterations'] < convergence_info[20]['avg_iterations'] else 20
        print(f"Grid size {faster_convergence} converges faster on average.")

        print("Larger grid sizes have more states, which generally increases the complexity and iterations needed for convergence.")

run_simulations()