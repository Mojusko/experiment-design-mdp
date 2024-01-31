import numpy as np
import matplotlib.pyplot as plt
from  experiments.asynch_benchmarks.fit_asynch_benchmarks import Branin2D, Michalewicz2D, Hartmann3D, Hartmann6D

func_num = 3
delay = 25
noise = 0.0001
episode_length = 100

if func_num == 1:
    func = Branin2D()
    func_idx = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 25]
    delta_mov = 0.05
elif func_num == 2:
    func = Michalewicz2D()
    func_idx = [6, 9, 22, 23, 25]
    delta_mov = 0.05
elif func_num == 3:
    func = Hartmann3D()
    func_idx = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    delta_mov = 0.1
elif func_num == 4:
    func = Hartmann6D()
    func_idx = []
    delta_mov = 0.2

num_features = int(2 ** (func.dim + 5))

# file paths
file_name_outer = f'experiments/asynch_benchmarks/results/' + func.name + f'/delay_{delay}/delta_mov_{delta_mov}/noise_var_{noise}/num_features_{num_features}/episode_length_{episode_length}/'
    
methods = ['/TruncatedSnAKe', '/MDPExplore/num_maximizers_100']
method_cols = ['r', 'y']
algo_label = ['TrSnAKe', 'MDP-BO']

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
    plt.plot(np.median(regret, axis = 0), label = algo_label[algo_idx])
    # plot quantiles
    plt.fill_between(np.arange(100), np.quantile(regret, 0.1, axis = 0), np.quantile(regret, 0.9, axis = 0), alpha = 0.2)

plt.legend()
print('stop')