import time
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import random
from matplotlib import pyplot as plt

def plot_values(V, P, dim):

    V_sq = np.reshape(V, (dim, dim))
    P_sq = np.reshape(P, (dim, dim))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    if dim < 10:
        fontSize = 20
    else:
        fontSize = 10

    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 2), ha='center', va='top', fontsize=fontSize)

        if np.round(label, 3) > -0.09  and P_sq[j][i] == 0:
            ax.text(i, j, 'LEFT', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 1:
            ax.text(i, j, 'DOWN', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 2:
            ax.text(i, j, 'RIGHT', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 3:
            ax.text(i, j, 'UP', ha='center', va='bottom', fontsize=fontSize)

    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    fig.tight_layout()
    plt.savefig('Images\\VI-FL-Plot_vals' + str(dim) + '.png')
    plt.show()

def eps_greedy(Q, s, eps=0.1):

    if np.random.uniform(0, 1) < eps:

        return np.random.randint(Q.shape[1])
    else:

        return greedy(Q, s)

def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    # Ensure s is an integer
    if not isinstance(s, int):
        s = s[0]

    return np.argmax(Q[s])

def run_episodes(env, Q, num_episodes=100):

    tot_rew = []

    for _ in range(num_episodes):
        done = False
        game_rew = 0
        state = env.reset()

        if not isinstance(state, int):
            state = state[0]

        while not done:
            action = greedy(Q, state)
            print(f"Action: {action}, Type: {type(action)}")  # Add this print statement

            if isinstance(action, (tuple, list)):
                action = action[0]

            step_output = env.step(action)
            print(f"Step Output: {step_output}")  # Add this print statement
            next_state, rew, done, extra_boolean, info = env.step(action)

            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    return np.mean(tot_rew)

def Q_learning(env, lr=0.01, lr_min=0.0001, lr_decay=0.99, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005,
               eps_min=0.0001):
    nA = env.action_space.n
    nS = env.observation_space.n

    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        if not isinstance(state, int):
            state = state[0]

        if eps > eps_min:
            eps *= eps_decay
            eps = max(eps, eps_min)

        if lr > lr_min:
            lr *= lr_decay
            lr = max(lr, lr_min)

        while not done:

            action = eps_greedy(Q, state, eps)

            next_state, rew, done, extra_boolean, info = env.step(action)

            if not isinstance(next_state, int):
                next_state = next_state[0]

            rew -= (0.01 * done)

            Q[state][action] = Q[state][action] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

    return Q

def run_fl(size):
    seed_val = 42
    np.random.seed(seed_val)
    random.seed(seed_val)
    if size == 4:
        env = gym.make("FrozenLake-v1")
    else:
        seed_val = 58
        np.random.seed(seed_val)
        random.seed(seed_val)
        dim = size
        random_map = generate_random_map(size=dim, p=0.8)

        env = gym.make("FrozenLake-v1", desc=random_map)

    env.reset()

    env = env.unwrapped

    learning_rates = [0.001, 0.01, 0.00001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    lr_decays = [1.0, 0.99, 0.9999, 0.999, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    lr_mins = [0.00001]
    epsilons = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    gammas = [0.9999, 0.999, 0.99, 0.9, 0.99999, 0.999999, 1.0]
    epsilon_decays = [0.99, 0.9999, 0.99999, 0.999999, 0.999, 0.9, 0.8, 0.7]
    epsilon_mins = [0.00001]

    best_lr, best_e, best_g, best_ed, best_em, best_rew = 0, 0, 0, 0, 0, -1

    for em in epsilon_mins:
        for am in lr_mins:
            for ad in lr_decays:
                for e in epsilons:
                    for g in gammas:
                        for a in learning_rates:
                            for ed in epsilon_decays:

                                tot_rew = 0
                                num_seeds = 4
                                cnt = 0
                                for x in range(num_seeds):
                                    cnt += 1
                                    seed_val = x
                                    np.random.seed(seed_val)
                                    random.seed(seed_val)
                                    Q_qlearning = Q_learning(env, lr=a, lr_decay=ad, lr_min=am,
                                                             num_episodes=1000, eps=e, gamma=g, eps_decay=ed,
                                                             eps_min=em)
                                    tot_rew += run_episodes(env, Q_qlearning, 15)
                                    if tot_rew < 0.3:
                                        break

                                print(e, em, ed, a, ad, am, g, tot_rew / cnt)

    iter_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,10000]
    time_arr = []
    rew_arr = []
    V_arr = []
    for t in tests:
        print(t)
        temp_iter = []
        for i in iter_arr:
            temp_iter.append(i)
            start = time.time()
            Q_qlearning = Q_learning(env, lr=t[3], lr_decay=t[4], lr_min=t[5],
                                     num_episodes=i, eps=t[0], gamma=t[6], eps_decay=t[2],
                                     eps_min=t[1])
            run_time = time.time() - start
            print('running tests')
            rew = run_episodes(env, Q_qlearning, 200) * 100
            tot_V = 0
            for s in range(env.observation_space.n):
                tot_V += Q_qlearning[s][np.argmax(Q_qlearning[s])]
            print(i, rew, tot_V / env.observation_space.n, run_time)
            time_arr.append(run_time)
            rew_arr.append(rew)
            V_arr.append((tot_V))

            fig, ax1 = plt.subplots()

            color = 'tab:blue'
            ax1.set_ylabel('Reward %/Avg V', color=color)
            ax1.plot(temp_iter, rew_arr, color=color, label='Reward %')
            ax1.plot(temp_iter, V_arr, color='darkblue', label='Avg V')
            ax1.legend()
            ax2 = ax1.twinx()

            color = 'tab:red'
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Time', color=color)
            ax2.plot(temp_iter, time_arr, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('V/Reward/Time vs. Iterations fpr 16x16 FL')
            plt.savefig('Images\\QL-Forest-16-RunStats' + str(size) + '.png')
            plt.show()

    def extract_policy(value_table, gamma=1.0):
        policy = np.zeros(env.observation_space.n)
        for state in range(env.observation_space.n):
            Q_table = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
            policy[state] = np.argmax(Q_table)

        return policy

    V = np.zeros(env.observation_space.n)
    P = np.zeros(env.observation_space.n)
    for s in enumerate(Q_qlearning):
        V[s[0]] = s[1][np.argmax(s[1])]
        P[s[0]] = np.argmax(s[1])

    plot_values(V, P, size)

run_fl(4)

run_fl(16)