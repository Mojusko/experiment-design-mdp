import numpy as np
import matplotlib.pyplot as plt

# arguments of the problem
noise = 0.0001
num_features = 121
episodes = 10
episode_length = 10
num_repetitions = 25
delta_mov = 0.15
episodic = True

if episodic:
    file_name = f'experiments/flow_ode_mono/results/episodic_feedback/delta_mov_{delta_mov}/noise_var_{noise}/num_features_{num_features}/num_episodes_{episodes}/episode_length_{episode_length}/'
else:
    file_name = f'experiments/flow_ode_mono/results/immediate_feedback/delta_mov_{delta_mov}/noise_var_{noise}/num_features_{num_features}/num_episodes_{episodes}/episode_length_{episode_length}/'

algo_names = ['EI', 'MDPExplore/policy_type_density/num_components_1', 'MDPExplore/policy_type_density/num_components_10', 'MDPExplore/policy_type_density/num_components_25']
algo_labels = ['MDP-EI', 'MDP-BO (1)', 'MDP-BO (10)', 'MDP-BO (25)']
# algo_names = ['EI', 'MDPExplore/policy_type_density/num_components_1']
# algo_labels = ['MDP-EI', 'MDP-BO (1)']
cols = ['blue', 'orange', 'green', 'red']

# get the real function
from experiments.flow_ode.fit_flow_ode import SchreckerODE
ode_solution = SchreckerODE()
theta_star = lambda x: ode_solution.solve(t_span = np.linspace(0, (x[:, 0].item() + 0.5) * 40), y0 = [0, 1 - (x[:, 1].item() + 0.5), x[:, 1].item() + 0.5, 0, 0, 0, 0])[:, 0][-1]
# obtain the optimal action
grid_1d = np.linspace(-0.5, 0.5, 11)
grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
action_space = grid
optimal_action = action_space[np.argmax([theta_star(act.reshape(-1, 2)) for act in action_space])].reshape(1, -1)
# obtain the optimal observation
optimal_observation = theta_star(optimal_action.reshape(1, -1))

# initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# initial regret
# init_regret = np.log(optimal_observation - theta_star(action_space[59].reshape(1, -1)) + 1e-8)
# init_regret = (optimal_observation - theta_star(action_space[59].reshape(1, -1))) / 8

# zero regret for plot
if episodic:
    zero_regret = np.zeros((len(algo_names), episodes))
else:
    zero_regret = np.zeros((len(algo_names), episodes * episode_length))

# initialize max regret
max_regret = 0
min_regret = 100

for algo_idx, algo in enumerate(algo_names):
    # load the data
    if episodic:
        best_guess = np.zeros((num_repetitions, episodes))
        regret = np.zeros((num_repetitions, episodes))
        for rep in range(0, num_repetitions):
            best_arms_rep = np.load(f'{file_name}{algo}/seed_{rep + 1}/best_guesses.npy')
            for arm_idx, arm in enumerate(best_arms_rep):
                best_guess[rep, arm_idx] = theta_star(arm.reshape(1, -1))
            
            # regret is the difference between the best arm and the optimal arm
            regret[rep] = optimal_observation - best_guess[rep] + 1e-7
        
    else:
        best_guess = np.zeros((num_repetitions, episodes * episode_length))
        regret = np.zeros((num_repetitions, episodes * episode_length))
        for rep in range(0, num_repetitions):
            best_arms_rep = np.load(f'{file_name}{algo}/seed_{rep + 1}/best_guesses.npy')
            for arm_idx, arm in enumerate(best_arms_rep):
                best_guess[rep, arm_idx] = theta_star(arm.reshape(1, -1))
            
            # regret is the difference between the best arm and the optimal arm
            regret[rep] = optimal_observation - best_guess[rep] + 1e-7

    # calculate the percentage of seeds where the regret is zero
    zero_regret[algo_idx] = np.sum(regret <= 1e-7, axis=0) / num_repetitions

    # plot the results, using the median and the 25th and 75th percentiles
    if episodic:
        mean_regret = np.mean(regret, axis=0)
        max_regret = np.maximum(np.max(mean_regret), max_regret)
        min_regret = np.minimum(np.min(mean_regret), min_regret)
        # mean_regret = np.append(init_regret, mean_regret)

        median_regret = np.median(regret, axis=0)
        # median_regret = np.append(init_regret, median_regret)

        lower_percentile = np.percentile(regret, 10, axis=0)
        # lower_percentile = np.append(init_regret, lower_percentile)

        upper_percentile = np.percentile(regret, 90, axis=0)
        # upper_percentile = np.append(init_regret, upper_percentile)
        ax.plot(np.arange(1, episodes + 1), mean_regret, color='k', linewidth = 5)
        ax.plot(np.arange(1, episodes + 1), mean_regret, label=algo_labels[algo_idx], markerfacecolor='black', marker='o', markersize=10, color=cols[algo_idx], linewidth = 3)
        # ax2.plot(range(0, episodes + 1), median_regret, label=algo, color=cols[algo_idx])
        # ax2.fill_between(np.arange(0, episodes + 1), lower_percentile, upper_percentile, alpha=0.2, color=cols[algo_idx])

    else:
        # take the log of the regret
        regret = np.log(regret)

        mean_regret = np.mean(regret, axis=0)
        max_regret = np.maximum(np.max(mean_regret), max_regret)
        min_regret = np.minimum(np.min(mean_regret), min_regret)
        # do line plots for the percentage of seeds where the regret is zero at the end of each episode
        episode_break_points = np.arange(episode_length - 1, episodes * episode_length, episode_length)

        # ax.plot(np.arange(1, episodes * episode_length + 1), mean_regret, color='black', linewidth = 5)
        # ax.plot(np.arange(1, episodes * episode_length + 1), mean_regret, label=algo_labels[algo_idx], color=cols[algo_idx], linewidth = 3)
        ax.plot(episode_break_points + 1, mean_regret[episode_break_points], color='black', linewidth = 5)
        ax.plot(episode_break_points + 1, mean_regret[episode_break_points], label=algo_labels[algo_idx], color=cols[algo_idx], linewidth = 3, markerfacecolor='black', marker='o', markersize=10)
        # ax2.plot(np.median(regret), label=algo, color=cols[algo_idx])
        # ax2.fill_between(np.arange(len(regret)), lower_percentile, upper_percentile, alpha=0.2, color=cols[algo_idx])


if episodic:
    ax.patch.set_visible(False)
    # do a group bar chart with the percentage of seeds where the regret is zero
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    # plot the group bar chart
    width = 0.2
    x = np.arange(1, episodes + 1)
    ax2.bar(x - 1.5 * width, zero_regret[0, :], width, label = algo_labels[0], color = cols[0])
    ax2.bar(x - 0.5 * width, zero_regret[1, :], width, label = algo_labels[1], color = cols[1])
    ax2.bar(x + 0.5 * width, zero_regret[2, :], width, label = algo_labels[2], color = cols[2])
    ax2.bar(x + 1.5 * width, zero_regret[3, :], width, label = algo_labels[3], color = cols[3])

    ax.set_xlabel('Episode', fontsize=20)
    ax.set_ylabel('Average Regret', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set y ticks at 0 and 0.035
    # calculate the maximum regret
    # max_regret = np.max(mean_regret)
    # reduce to 3 decimal places
    max_regret = np.round(max_regret, 3)
    ax.set_yticks([0, max_regret])
    ax.set_xticks(np.arange(1, episodes + 1))

    # flip the labels for the second axis
    ax2.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', which='major', labelsize=15)
    # set y ticks at 0, 0.5 and 1
    ax2.set_yticks([0, 0.5, 1])

    ax.legend(loc='upper right', fontsize=15)

else:
    ax.patch.set_visible(False)

    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Average Log Regret', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # calculate the maximum regret
    # reduce to 2 decimal places
    max_regret = np.round(max_regret, 0)
    min_regret = np.round(min_regret, 0)
    ax.set_yticks([min_regret, max_regret])
    ax.set_xticks(np.arange(1, episodes + 1) * 10)
    
    ax.legend(loc='upper right', fontsize=15)

    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax2.patch.set_visible(False)
    ax2.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', which='major', labelsize=15)
    
    # do line plots for the percentage of seeds where the regret is zero at the end of each episode
    episode_break_points = np.arange(episode_length - 1, episodes * episode_length, episode_length)
    width = 1

    # ax2.bar(episode_break_points + 1 - 0.5 * width, zero_regret[0, episode_break_points], width, label = algo_labels[0], color = cols[0])
    # ax2.bar(episode_break_points + 1 + 0.5 * width, zero_regret[1, episode_break_points], width, label = algo_labels[1], color = cols[1])
    ax2.bar(episode_break_points + 1 - 1.5 * width, zero_regret[0, episode_break_points], width, label = algo_labels[0], color = cols[0])
    ax2.bar(episode_break_points + 1 - 0.5 * width, zero_regret[1, episode_break_points], width, label = algo_labels[1], color = cols[1])
    ax2.bar(episode_break_points + 1 + 0.5 * width, zero_regret[2, episode_break_points], width, label = algo_labels[2], color = cols[2])
    ax2.bar(episode_break_points + 1 + 1.5 * width, zero_regret[3, episode_break_points], width, label = algo_labels[3], color = cols[3])

# save the figure
if episodic:
    fig.savefig(f'flow_ode_mono_episodic_results.png', bbox_inches='tight', dpi=300)
else:
    fig.savefig(f'flow_ode_mono_immediate_results.png', bbox_inches='tight', dpi=300)

plt.show()