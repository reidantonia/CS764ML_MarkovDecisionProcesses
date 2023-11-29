import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn as sns

RANDOM_SEED = 20
NUM_RUNS = 100

def initialize_environment(lake_size, map_seed=RANDOM_SEED):
    random.seed(map_seed)
    lake_map = generate_random_map(size=lake_size, p=0.8) if lake_size != 4 else None
    environment = gym.make("FrozenLake-v1", desc=lake_map)
    return environment.unwrapped

def create_heatmap(values, grid_size, title, filename_suffix, run, discount_factor):
    text_fontsize = 6


    reshaped_values = np.reshape(values, (grid_size, grid_size))
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(reshaped_values, cmap="coolwarm", interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('State Values', rotation=-90, va="bottom")
    for (j, i), val in np.ndenumerate(reshaped_values):
        ax.text(i, j, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black',
                fontsize=text_fontsize)
    row_labels = col_labels = [str(i) for i in range(grid_size)]
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title(title)
    plt.savefig(f'Images/FrozenLake_VI_Size{grid_size}_DF{discount_factor}_Values{run}_{filename_suffix}.png')
    plt.close()

def plot_policy(V, policy, size, filename_suffix, discount_factor):
    V_sq = np.reshape(V, (size, size))
    P_sq = np.reshape(policy, (size, size))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(V_sq, cmap="coolwarm", interpolation='nearest')
    fontSize = 20 if size < 10 else 10
    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 2), ha='center', va='center', fontsize=fontSize)
        action = ['LEFT', 'DOWN', 'RIGHT', 'UP'][P_sq[j][i]]
        ax.text(i, j, action, ha='center', va='bottom', fontsize=fontSize)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('State-Value Function and Policy')
    plt.savefig(f'Images/FrozenLake_VI_Size{size}_DF{discount_factor}_{filename_suffix}.png')
    plt.close()

def perform_value_iteration(environment, discount_factor=0.99, convergence_threshold=0.0001):
    state_values = np.zeros(environment.observation_space.n)
    value_history = []
    max_values = []
    mean_values = []
    while True:
        max_value_change = 0
        for state in range(environment.observation_space.n):
            old_value = state_values[state]
            action_values = [sum([prob * (reward + discount_factor * state_values[next_state])
                                  for prob, next_state, reward, _ in environment.P[state][action]])
                             for action in range(environment.action_space.n)]
            state_values[state] = np.max(action_values)
            max_value_change = max(max_value_change, abs(old_value - state_values[state]))
        value_history.append(state_values.copy())
        max_values.append(np.max(state_values))
        mean_values.append(np.mean(state_values))
        if max_value_change < convergence_threshold:
            break
    policy = np.zeros(environment.observation_space.n, dtype=int)
    for state in range(environment.observation_space.n):
        action_values = [sum([prob * (reward + discount_factor * state_values[next_state])
                              for prob, next_state, reward, _ in environment.P[state][action]])
                         for action in range(environment.action_space.n)]
        policy[state] = np.argmax(action_values)
    return state_values, policy, value_history, max_values, mean_values

def plot_convergence(value_history, grid_size, run, suffix, discount_factor):
    policy_changes = [np.sum(value_history[i] != value_history[i + 1]) for i in range(len(value_history) - 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(policy_changes)
    plt.title(f'Policy Convergence Over Iterations for Size-{grid_size} Frozen Lake')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Policy Changes')
    plt.grid(True)
    plt.savefig(f'Images/FrozenLake_ValueIteration_{grid_size}_DF{discount_factor}_{run}_{suffix}.png')
    plt.close()

def visualize_detailed_convergence(max_values, mean_values, grid_size, run_number, discount_factor):
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
    plt.savefig(f'Images/FrozenLake_VI_Conv_Size{grid_size}_DF{discount_factor}_Run{run_number}.png')
    plt.close()

def run_simulations():
    discount_factors = [0.9, 0.99, 0.999]
    for grid_size in [20]:

    # for grid_size in [5, 20]:
        for discount_factor in discount_factors:
            cumulative_values = []
            iteration_counts = []
            for run in range(NUM_RUNS):
                env = initialize_environment(grid_size)
                values, policy, value_history, max_values, mean_values = perform_value_iteration(env, discount_factor)
                cumulative_values.append(values)
                iteration_counts.append(len(value_history))
                create_heatmap(values, grid_size, f'State Values for {grid_size}x{grid_size} Grid', 'heatmap', run, discount_factor)
                # visualize_detailed_convergence(max_values, mean_values, grid_size, run, discount_factor)
                # plot_convergence(value_history, grid_size, run, 'value_it_conv', discount_factor)

            avg_iterations = np.mean(iteration_counts)
            max_v_all_runs = np.max([np.max(vals) for vals in cumulative_values])
            mean_v_all_runs = np.mean([np.mean(vals) for vals in cumulative_values])

            print(f"\nModel: Frozen Lake")
            print(f"Size: {grid_size}, Discount Factor: {discount_factor}")
            print(f"Iterations to Converge: {avg_iterations}")
            print(f"Max V (across all runs): {max_v_all_runs}")
            print(f"Mean V (across all runs): {mean_v_all_runs}")

run_simulations()