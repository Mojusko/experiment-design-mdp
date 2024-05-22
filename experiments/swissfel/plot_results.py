import numpy as np
import matplotlib.pyplot as plt

# obtain the files
file_name_outer = 'experiments/swissfel/results/'

algo_list = ['mdp', 'worst']
algo_label = ['MDP-BO', 'State-less BO']
method_cols = ['orange', 'blue']

fig, ax = plt.subplots(figsize=(8, 5))
# initialize the second axis
ax.patch.set_visible(False)
ax2 = ax.twinx()
ax2.set_zorder(ax.get_zorder() - 1)

max_regret = 0
min_regret = 0

for algo_idx, algo in enumerate(algo_list):
    regret = np.zeros((25, 100))
    for seed_idx in range(1, 26):
        file_name = file_name_outer + algo + f'/seed_{seed_idx}/horizon_100/'
        # load the results
        regret[seed_idx - 1, :] = np.load(file_name + 'regrets.npy')

    # plot median
    # ax.plot(np.median(regret, axis = 0), label = algo_label[algo_idx], color = method_cols[algo_idx], linewidth = 5)
    # plot quantiles
    # ax.fill_between(np.arange(100), np.quantile(regret, 0.1, axis = 0), np.quantile(regret, 0.9, axis = 0), alpha = 0.2, color = method_cols[algo_idx])
    
    # duplicate the last column of regret
    regret = np.concatenate([regret, regret[:, -1].reshape(-1, 1)], axis = 1)

    # plot the mean regret
    mean_regret = np.mean(regret, axis = 0)
    # duplicate the last element
    # mean_regret = np.append(mean_regret, mean_regret[-1])
    # plot every tenth point
    ax.plot(np.arange(0, 101, 10), mean_regret[::10], color='k', linewidth = 5)
    ax.plot(np.arange(0, 101, 10), mean_regret[::10], label=algo_label[algo_idx], markerfacecolor='black', marker='o', markersize=10, color=method_cols[algo_idx], linewidth = 3)
    # ax.plot(np.arange(1, episodes + 1), mean_regret, label=algo_label[algo_idx], markerfacecolor='black', marker='o', markersize=10, color=method_cols[algo_idx], linewidth = 3)

    # obtain the percentage of seeds where the regret is zero
    zero_regret = np.mean(regret <= 1e-7, axis=0)
    x = np.arange(0, 101, 10)
    width = 2.5
    if algo_idx == 0:
        ax2.bar(x - 0.5 * width, zero_regret[::10], color = method_cols[algo_idx], width = width, label = algo_label[algo_idx])
    else:
        ax2.bar(x + 0.5 * width, zero_regret[::10], color = method_cols[algo_idx], width = width, label = algo_label[algo_idx])

    max_regret = np.maximum(max_regret, np.max(mean_regret[::10]))

ax.legend(fontsize = 15)

# set labels
ax.set_xlabel('Iteration', fontsize = 20)
ax.set_ylabel('Average Regret', fontsize = 20)
# set tick size
ax.tick_params(axis='both', which='major', labelsize=15)
# set y ticks
# max_regret = np.round(max_regret, 0)
ax.set_yticks([0, max_regret])

# set labels for the second axis
ax2.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
ax2.set_ylim([0, 1])
ax2.tick_params(axis='y', which='major', labelsize=15)
# set y ticks at 0, 0.5 and 1
ax2.set_yticks([0, 0.5, 1])

# save figure
plt.savefig('swissfel.png', bbox_inches='tight', dpi = 300)

plt.show()