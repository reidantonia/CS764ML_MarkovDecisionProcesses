import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn

RANDOM_SEED = 20
NUM_RUNS = 100

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

    plt.savefig(f'Images/FrozenLake_PI_Size{grid_size}_Values{run}_{filename_suffix}.png')
    plt.close()

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
    plt.savefig(f'Images/FrozenLake_PI_Size{size}_{filename_suffix}.png')
    plt.close()

def perform_policy_iteration(environment, discount_factor=0.99, convergence_threshold=0.0001, track_detailed=False):
    state_values = np.zeros(environment.observation_space.n)
    current_policy = np.random.choice(environment.action_space.n, environment.observation_space.n)
    policy_history = []
    value_history = [] if track_detailed else None

    while True:
        is_policy_stable = True
        policy_history.append(current_policy.copy())

        while True:
            max_value_change = 0
            for state in range(environment.observation_space.n):
                old_value = state_values[state]
                state_values[state] = sum([prob * (reward + discount_factor * state_values[next_state])
                                           for prob, next_state, reward, _ in
                                           environment.P[state][current_policy[state]]])
                max_value_change = max(max_value_change, abs(old_value - state_values[state]))
            if max_value_change < convergence_threshold:
                break

        if track_detailed:
            value_history.append(state_values.copy())

        for state in range(environment.observation_space.n):
            old_action = current_policy[state]
            action_values = [sum([prob * (reward + discount_factor * state_values[next_state])
                                  for prob, next_state, reward, _ in environment.P[state][action]])
                             for action in range(environment.action_space.n)]
            current_policy[state] = np.argmax(action_values)
            if old_action != current_policy[state]:
                is_policy_stable = False

        if is_policy_stable:
            break

    if track_detailed:

        max_values = [np.max(V) for V in value_history]
        mean_values = [np.mean(V) for V in value_history]
        return state_values, current_policy, policy_history, max_values, mean_values

    return state_values, current_policy, policy_history

def visualize_results(values, policy, grid_size, suffix):
    reshaped_values = np.reshape(values, (grid_size, grid_size))
    reshaped_policy = np.reshape(policy, (grid_size, grid_size))

    action_symbols = ['←', '↓', '→', '↑']

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(reshaped_values, cmap='coolwarm',alpha=0.5)

    for (j, i), value in np.ndenumerate(reshaped_values):
        action = action_symbols[reshaped_policy[j, i]]
        text_color = 'black' if value > 0.5 else 'white'
        ax.text(i, j, f"{action}\n{value:.2f}", ha="center", va="center", color=text_color,
                fontsize=14 if grid_size <= 8 else 8)

    ax.set_title(f'State-Value Function and Policy for Size-{grid_size} Frozen Lake', fontsize=16, pad=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('State Values', rotation=-90, va="bottom")

    plt.savefig(f"Images/FrozenLake_PI_Size{grid_size}_{suffix}.png")
    plt.close()

def plot_policy_convergence(policy_history, grid_size, run, suffix):
    policy_changes = [np.sum(policy_history[i] != policy_history[i + 1]) for i in range(len(policy_history) - 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(policy_changes)
    plt.title(f'Policy Convergence Over Iterations for Size-{grid_size} Frozen Lake')
    plt.xlabel('Iteration')

    plt.ylabel('Number of Policy Changes')
    plt.grid(True)
    plt.savefig(f'Images/FrozenLake_PolicyConvergence_{grid_size}_{run}_{suffix}.png')
    plt.close()

def visualize_detailed_convergence(max_values, mean_values, grid_size, run_number):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(max_values, label='Max V')
    plt.xlabel('Iteration')
    plt.ylabel('Max Value')
    plt.title(f'Max V per Iteration for Size-{grid_size} Frozen Lake')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mean_values, label='Mean V')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Value')
    plt.title(f'Mean V per Iteration for Size-{grid_size} Frozen Lake')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Images/FrozenLake_PI_Conv_Size{grid_size}_Run{run_number}.png')
    plt.close()

def run_simulations():
    convergence_info = {}
    for grid_size in [5, 20]:
        cumulative_values = []
        cumulative_policies = []
        iteration_counts = []

        for run in range(NUM_RUNS):
            env = initialize_environment(grid_size)
            values, policy, policy_history, max_values, mean_values = perform_policy_iteration(env, track_detailed=True)

            cumulative_values.append(values)
            cumulative_policies.append(policy)
            iteration_counts.append(len(policy_history))

            create_heatmap(values, grid_size, f'State Values for {grid_size}x{grid_size} Grid', 'heatmap', run)
            plot_policy_convergence(policy_history, grid_size, run, 'policy_cov')
            visualize_detailed_convergence(max_values, mean_values, grid_size, run)

        avg_iterations = np.mean(iteration_counts)
        convergence_info[grid_size] = {
            'avg_iterations': avg_iterations,
            'values': np.mean(cumulative_values, axis=0),
            'policies': cumulative_policies
        }

        print(f"Size {grid_size} - Average iterations to converge: {avg_iterations}")
        print(f"Size {grid_size} - Average Max V: {np.max(convergence_info[grid_size]['values'])}, Average Mean V: {np.mean(convergence_info[grid_size]['values'])}")

    if all(size in convergence_info for size in [5, 20]):
        print("\nConvergence Comparison:")
        faster_convergence = 5 if convergence_info[5]['avg_iterations'] < convergence_info[10]['avg_iterations'] else 10
        print(f"Grid size {faster_convergence} converges faster on average.")

        same_policy = all(
            np.array_equal(convergence_info[5]['policies'][i], convergence_info[20]['policies'][i]) for i in
            range(NUM_RUNS))
        print(f"Do they converge to the same answer? {'Yes' if same_policy else 'No'}")

        print("Larger grid sizes have more states, which generally increases the complexity and iterations needed for convergence.")

run_simulations()