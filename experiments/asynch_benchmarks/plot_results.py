import numpy as np
import matplotlib.pyplot as plt
from  experiments.asynch_benchmarks.fit_asynch_benchmarks import Branin2D, Michalewicz2D, Hartmann3D, Hartmann6D, Levy4D, Michalewicz3D

func_num = 6
delay = 25
noise = 0.0001
episode_length = 100

if func_num == 1:
    func = Branin2D()
    func_idx = list(range(1, 26))
    delta_mov = 0.05
elif func_num == 2:
    func = Michalewicz2D()
    func_idx = list(range(1, 26))
    delta_mov = 0.05
elif func_num == 3:
    func = Hartmann3D()
    func_idx = list(range(1, 26))
    delta_mov = 0.1
elif func_num == 4:
    func = Hartmann6D()
    func_idx = list(range(1, 26))
    # func_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    delta_mov = 0.2
elif func_num == 5:
    func = Levy4D()
    func_idx = list(range(1, 26))
    delta_mov = 0.1
elif func_num == 6:
    func = Michalewicz3D()
    func_idx = list(range(1, 26))
    delta_mov = 0.1

num_features = np.minimum(int(2 ** (func.dim + 5)), 512)

# file paths
file_name_outer = f'experiments/asynch_benchmarks/results/' + func.name + f'/delay_{delay}/delta_mov_{delta_mov}/noise_var_{noise}/num_features_{num_features}/episode_length_{episode_length}/'
    
methods = ['MDPExplore/thompson_sampling/num_maximizers_100', 'MDPExplore/ucb/num_maximizers_25', 'TruncatedSnAKe']
method_cols = ['orange', 'green', 'purple']
algo_label = ['MDP-BO-TS', 'MDP-BO-UCB', 'TrSnAKe']

# methods = ['MDPExplore/thompson_sampling/num_maximizers_100', 'MDPExploreGDesign/thompson_sampling/num_maximizers_100']
# method_cols = ['orange', 'red']
# algo_label = ['MDP-XY-BO-TS (100)', 'MDP-G-BO-TS (100)']

for algo_idx, algo_name in enumerate(methods):
    regret = np.zeros((len(func_idx), 100))
    for seed_idx, seed in enumerate(func_idx):
        file_name = file_name_outer + algo_name + f'/seed_{seed}/'
        # load the results
        x_evaluations = np.load(file_name + 'x_evaluations.npy')
        f_evaluations = np.load(file_name + 'f_evaluations.npy')
        best_guesses = np.load(file_name + 'best_guesses.npy')
        # obtain the regret
        for t in range(100):
            f_guess = func.query_function(best_guesses[t].reshape(1, -1))
            regret[seed_idx, t] = func.optimum - f_guess
            # regret[seed_idx, t] = func.optimum - np.max(f_evaluations[:t+1])
        
    # plot the results
    plt.plot(np.median(regret, axis = 0), label = algo_label[algo_idx], color = method_cols[algo_idx], linewidth = 5)
    # plot quantiles
    plt.fill_between(np.arange(100), np.quantile(regret, 0.1, axis = 0), np.quantile(regret, 0.9, axis = 0), alpha = 0.2, color = method_cols[algo_idx])

# set x limits
plt.xlim([20, 100])
plt.legend(fontsize = 14)
# set y limits depending on the function
if func_num == 1:
    plt.ylim([-0.05, 3.05])
elif func_num == 2:
    pass
elif func_num == 3:
    plt.ylim([-0.05, 3.5])
else:
    pass
# set labels
plt.xlabel('Iteration', fontsize = 14)
plt.ylabel('Median Regret', fontsize = 14)
# set tick size
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

# set figure size
fig = plt.gcf()
fig.set_size_inches(6, 2.5)

# save the figure
title = f'{func.name}_asynch_experiment.png'
# title = f'{func.name}_asynch_experiment_main.png'
# title = f'{func.name}_ucb_ablation_asynch_experiment.png'
plt.savefig(title, bbox_inches='tight', dpi = 300)

print('stop')