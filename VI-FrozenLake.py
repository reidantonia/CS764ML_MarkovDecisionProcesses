import numpy as np
import gym
from matplotlib import pyplot as plt
import time
import matplotlib
from gym.envs.toy_text.frozen_lake import generate_random_map
import random

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

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.title('epsilon')
    plt.ylabel('gamma')

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def eval_state_action(env, V, s, a, gamma=0.99):
    return np.sum([p * ((rew - 0.01 * _) + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def value_iteration(env, nA, nS, epsilon=0.0001, gamma=0.99):
    V = np.zeros(nS)
    it = 0
    delta_vals = []
    avg_V = []
    while True:
        delta = 0

        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(env, V, s, a, gamma) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))

        delta_vals.append(delta)
        avg_V.append(np.mean(V))
        if delta < epsilon:
            break

        it += 1
    return V, delta_vals, it, avg_V

def run_vi_episodes(env, V, nA, num_games=100, gamma=0.99):
    tot_rew = 0
    state = env.reset()
    max_run = 0

    for _ in range(num_games):
        done = False

        while not done:

            action = np.argmax([eval_state_action(env, V, state, a, gamma) for a in range(nA)])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            tot_rew += reward
            max_run += 1
            if done:
                state = env.reset()
            if max_run > 10000:
                state = env.reset()
                done = True
                tot_rew += 0
                max_run = 0

    return float(tot_rew / num_games)

def run_fl(size):
    seed_val = 42
    np.random.seed(seed_val)
    random.seed(seed_val)
    if size == 4:
        env = gym.make("FrozenLake-v0")
    else:
        seed_val = 58
        np.random.seed(seed_val)
        random.seed(seed_val)
        dim = size
        random_map = generate_random_map(size=dim, p=0.8)

        env = gym.make("FrozenLake-v0", desc=random_map)
    env.seed(seed_val)
    env.reset()

    env = env.unwrapped

    nA = env.action_space.n
    nS = env.observation_space.n

    V = np.zeros(nS)
    policy = np.zeros(nS)

    best_delta = []
    best_V = np.zeros(nS)
    best_won = -1
    best_pol = np.zeros(nS)

    gammas = [0.1, 0.3, 0.4, 0.7, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1]
    epsilons = [0.1,
                0.01,
                0.001,
                0.0001,
                0.00001,
                0.000001,
                0.0000001,
                0.00000001,
                0.000000001,
                0.0000000001]

    gammas = [0.9]
    epsilons = [0.000001]

    per_won_hm = np.zeros((len(gammas), len(epsilons)))
    iters_hm = np.zeros((len(gammas), len(epsilons)))
    time_hm = np.zeros((len(gammas), len(epsilons)))

    g_cnt = 0
    e_cnt = 0
    for g in gammas:
        e_cnt = 0
        for e in epsilons:
            if g >= 0.99 and e <= 0.01:
                per_won_hm[g_cnt][e_cnt] = 0
                iters_hm[g_cnt][e_cnt] = 0
                time_hm[g_cnt][e_cnt] = 0
            else:
                start = time.time()
                V, delta_vals, iterations, avg_V = value_iteration(env, nA, nS, epsilon=e, gamma=g)

                run_time = time.time() - start

                per_won = 0
                per_won_hm[g_cnt][e_cnt] = per_won
                iters_hm[g_cnt][e_cnt] = iterations
                time_hm[g_cnt][e_cnt] = run_time * 1000
                print(g, e, iterations, per_won)
                if per_won > best_won:
                    best_delta = delta_vals
                    best_V = V
                    best_won = per_won
                    best_e = e
                    best_g = g
            e_cnt += 1
        g_cnt += 1

    print(best_e, best_g)
    print(avg_V)

    fig, ax = plt.subplots()

    im, cbar = heatmap(per_won_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="% Games Won")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.savefig('Images\\VI-FL-Per_Heatmap' + str(size) + '.png')
    plt.show()

    fig, ax = plt.subplots()

    im, cbar = heatmap(iters_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="# of Iterations to Convergence")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images\\VI-FL-Iter_Heatmap' + str(size) + '.png')
    plt.show()

    fig, ax = plt.subplots()

    im, cbar = heatmap(time_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="Runtime (ms)")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images\\VI-FL-Time_Heatmap' + str(size) + '.png')
    plt.show()

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('Max Delta', color=color)
    ax1.semilogy(delta_vals, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Avg V', color=color)
    ax2.semilogy(avg_V, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Delta/V vs. Iterations')
    plt.savefig('Images\\VI-FL-Delta_Vals' + str(size) + '.png')
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

    print(best_e, best_g, best_won)
    optimal_policy = extract_policy(V, gamma=best_g)

    print(V.reshape((size, size)))
    print(optimal_policy.reshape((size, size)))

    plot_values(V, optimal_policy, size)

run_fl(4)
run_fl(16)