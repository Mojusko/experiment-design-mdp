import numpy as np
import matplotlib.pyplot as plt

# obtain the files
file_name_outer = 'experiments/swissfel/results/'

algo_list = ['mdp', 'worst']
algo_label = ['MDP-BO', 'State-less BO']
method_cols = ['darkorange', 'blue']

for algo_idx, algo in enumerate(algo_list):
    regret = np.zeros((25, 100))
    for seed_idx in range(1, 26):
        file_name = file_name_outer + algo + f'/seed_{seed_idx}/horizon_100/'
        # load the results
        regret[seed_idx - 1, :] = np.load(file_name + 'regrets.npy')

    # plot median
    plt.plot(np.median(regret, axis = 0), label = algo_label[algo_idx], color = method_cols[algo_idx], linewidth = 5)
    # plot quantiles
    plt.fill_between(np.arange(100), np.quantile(regret, 0.1, axis = 0), np.quantile(regret, 0.9, axis = 0), alpha = 0.2, color = method_cols[algo_idx])

plt.legend(fontsize = 14)

# set labels
plt.xlabel('Iteration', fontsize = 14)
plt.ylabel('Inference Regret', fontsize = 14)
# set tick size
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
# set figure size
fig = plt.gcf()
fig.set_size_inches(7, 5)
# save figure
plt.savefig('swissfel.png', bbox_inches='tight', dpi = 300)

plt.show()