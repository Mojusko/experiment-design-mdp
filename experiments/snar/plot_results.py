import numpy as np
import matplotlib.pyplot as plt
from experiments.snar.fit_snar import SnAr

func = SnAr()
theta_star = lambda x: func.query_function(x.reshape(-1, 4))

delta_mov = 0.1
noise = 0.001
num_features = 512
episode_length = 100

algo_names = ['MDPExplore/num_maximizers_100', 'MDPExplore/ucb/num_maximizers_25', 'TruncatedSnAKe', 'LSR/gamma_0.01']
algo_labels = ['MDP-BO-TS (100)','MDP-BO-UCB (25)', 'TrSnAKe', 'LSR']
cols = ['orange', 'green', 'purple', 'blue']

file_name_outer = f'experiments/snar/results/SnarBenchmark/delta_mov_{delta_mov}/noise_var_{noise}/num_features_{num_features}/episode_length_{episode_length}/'

# plot the results
# fig, ax = plt.subplots(figsize=(8, 6))

pre_load_regret = False

for algo_idx, algo_name in enumerate(algo_names):
    if not pre_load_regret:
        regret = np.zeros((25, 100))
        for seed_idx, seed in enumerate(range(1, 26)):
            file_name = file_name_outer + algo_name + f'/seed_{seed}/'
            # load the results
            x_evaluations = np.load(file_name + 'x_evaluations.npy')
            f_evaluations = np.load(file_name + 'f_evaluations.npy')
            best_guesses = np.load(file_name + 'best_guesses.npy')
            # obtain the regret
            for t in range(100):
                f_guess = theta_star(best_guesses[t].reshape(1, -1))
                regret[seed_idx, t] = func.optimum - f_guess
                # regret[seed_idx, t] = func.optimum - np.max(f_evaluations[:t+1])
            
        # save the regret for quick future plotting
        np.save(file_name + 'predictive_regret.npy', regret)
    else:
        file_name = file_name_outer + algo_name + f'/seed_{25}/'
        regret = np.load(file_name + 'predictive_regret.npy')
        
    # plot the results
    plt.plot(np.median(regret, axis = 0), label = algo_labels[algo_idx], color = cols[algo_idx], linewidth = 5)
    # plot quantiles
    plt.fill_between(np.arange(100), np.quantile(regret, 0.1, axis = 0), np.quantile(regret, 0.9, axis = 0), alpha = 0.2, color = cols[algo_idx])

# set x and y labels
plt.xlabel('Iteration', fontsize = 14)
plt.ylabel('Median Regret', fontsize = 14)
# increase tick font size
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.legend(fontsize = 14)

# set figure size
fig = plt.gcf()
fig.set_size_inches(6, 3)

# save the figure
plt.savefig('snar_synch_experiment.png', bbox_inches='tight', dpi = 300)

plt.show()