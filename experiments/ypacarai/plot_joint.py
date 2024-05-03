import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# initialize the figure
fig, ax = plt.subplots(figsize=(16, 5), ncols=2)
ax = np.flip(ax)

# read the image and convert it to grayscale using Image package
lake = Image.open('ypacarai_lake.jpg').convert('L')

# convert the image to a numpy array
lake = np.asarray(lake)

# turn into a binary image
lake = (lake < 128).astype(int)
# rotate the image three times
lake = np.rot90(lake, 3)

# define the grid of actions
grid1 = np.linspace(-0.5, 0.5, lake.shape[0])
grid2 = np.linspace(-0.5, 0.5, lake.shape[1])
grid = np.array(np.meshgrid(grid1, grid2)).T.reshape(-1, 2)

# obtain the action space
lake_scatter = grid[lake.flatten() == 1]

# load the action space
ypacarai_action_space = np.load('ypacarai_centroids.npy')

# load the adjacency matrix
ypacarai_adjacency_matrix = np.load('ypacarai_adjacency_matrix.npy')

# make the background green
ax[0].set_facecolor('darkseagreen')

# plot the lake first as a scatter plot
ax[0].scatter(lake_scatter[:, 1], lake_scatter[:, 0], s=0.1)

# plot the centroids dark orange
ax[0].scatter(ypacarai_action_space[:, 1], ypacarai_action_space[:, 0], s=30, c='darkorange')

# define the initial state
init_state = 59
# define the terminal state
terminal_state = 43
optimal_state = 95
# plot a diamond near the initial state and a square near the terminal state
ax[0].scatter(ypacarai_action_space[init_state, 1], ypacarai_action_space[init_state, 0], marker='s', s=150, c='k', label='initial state')
# plt.scatter(ypacarai_action_space[terminal_state, 0], ypacarai_action_space[terminal_state, 1], marker='s', s=100, c='k', label='terminal state')
ax[0].scatter(ypacarai_action_space[95, 1], ypacarai_action_space[95, 0], marker='*', s=150, c='darkorange', label='global optima', zorder = 10)
ax[0].scatter(ypacarai_action_space[95, 1], ypacarai_action_space[95, 0], marker='*', s=500, c='k', zorder = 9)

ax[0].scatter(ypacarai_action_space[53, 1], ypacarai_action_space[53, 0], marker='s', s=100, c='darkorange', label='local optima', zorder = 10)
ax[0].scatter(ypacarai_action_space[53, 1], ypacarai_action_space[53, 0], marker='s', s=200, c='k', zorder = 9)

# create a line between the centroids if they are connected
for i in range(ypacarai_action_space.shape[0]):
    for j in range(ypacarai_action_space.shape[0]):
        if ypacarai_adjacency_matrix[i, j] == 1:
            ax[0].plot([ypacarai_action_space[i, 1], ypacarai_action_space[j, 1]], [ypacarai_action_space[i, 0], ypacarai_action_space[j, 0]], c = 'black')

# now load a trajectory chosen by the algorithm
file_name = 'experiments/ypacarai/results/episodic_feedback/noise_var_0.001/num_features_100/num_episodes_5/episode_length_50/MDPExplore/policy_type_density/num_components_1/seed_3'

x_evals = np.load(file_name + '/x_evaluations.npy')

# plot the x_eval trajectory in red
ax[0].plot(x_evals[:50, 1], x_evals[:50, 0], c='k', linewidth=6)
ax[0].plot(x_evals[:50, 1], x_evals[:50, 0], c='red', linewidth=3, label='chosen trajectory')

# remove xticks and yticks
ax[0].set_xticks([])
ax[0].set_yticks([])

# rotate the whole plot 90 degrees
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()

# plot legend in top right
ax[0].legend(loc='upper right', fontsize=14)

# arguments of the problem
noise = 0.001
num_features = 100
episodes = 5
episode_length = 50
num_repetitions = 25
episodic = True

if episodic:
    file_name = f'experiments/ypacarai/results/episodic_feedback/noise_var_{noise}/num_features_{num_features}/num_episodes_{episodes}/episode_length_{episode_length}/'
else:
    file_name = f'experiments/ypacarai/results/immediate_feedback/noise_var_{noise}/num_features_{num_features}/num_episodes_{episodes}/episode_length_{episode_length}/'

algo_names = ['EI', 'MDPExplore/policy_type_density/num_components_1', 'MDPExplore/policy_type_density/num_components_10', 'MDPExplore/policy_type_density/num_components_25']
algo_labels = ['MDP-EI', 'MDP-BO (1)', 'MDP-BO (10)', 'MDP-BO (25)']
# algo_names = ['EI', 'MDPExplore/policy_type_density/num_components_1', 'MDPExplore/policy_type_density/num_components_10']
# algo_labels = ['MDP-EI', 'MDP-B0 (1)', 'MDP-BO (10)']
cols = ['blue', 'orange', 'green', 'red']

# get the real function
from experiments.ypacarai.fit_ypacarai import Schekel2D
scheckel_function = Schekel2D()
theta_star = lambda x: scheckel_function.query_function(x.reshape(-1, 2))
# obtain the optimal action
action_space = np.load('ypacarai_centroids.npy')
optimal_action = action_space[np.argmax(theta_star(action_space))]
# obtain the optimal observation
optimal_observation = theta_star(optimal_action.reshape(1, -1))


# initial regret
# init_regret = np.log(optimal_observation - theta_star(action_space[59].reshape(1, -1)) + 1e-8)
init_regret = (optimal_observation - theta_star(action_space[59].reshape(1, -1))) / 8

# zero regret for plot
if episodic:
    zero_regret = np.zeros((len(algo_names), episodes))
else:
    zero_regret = np.zeros((len(algo_names), episodes * episode_length))

for algo_idx, algo in enumerate(algo_names):
    # load the data
    if episodic:
        best_guess = np.zeros((num_repetitions, episodes))
        regret = np.zeros((num_repetitions, episodes))
        for rep in range(0, num_repetitions):
            best_arms_rep = np.load(f'{file_name}{algo}/seed_{rep + 1}/best_guesses.npy')
            best_guess[rep] = theta_star(best_arms_rep)
            
            # regret is the difference between the best arm and the optimal arm
            regret[rep] = optimal_observation - best_guess[rep] + 1e-8
        
    else:
        best_guess = np.zeros((num_repetitions, episodes * episode_length))
        regret = np.zeros((num_repetitions, episodes * episode_length))
        for rep in range(0, num_repetitions):
            best_arms_rep = np.load(f'{file_name}{algo}/seed_{rep + 1}/best_guesses.npy')
            best_guess[rep] = theta_star(best_arms_rep)
            
            # regret is the difference between the best arm and the optimal arm
            regret[rep] = optimal_observation - best_guess[rep] + 1e-7

    # calculate the percentage of seeds where the regret is zero
    zero_regret[algo_idx] = np.sum(regret <= 1e-7, axis=0) / num_repetitions

    # plot the results, using the median and the 25th and 75th percentiles
    if episodic:
        mean_regret = np.mean(regret, axis=0)
        # mean_regret = np.append(init_regret, mean_regret)

        median_regret = np.median(regret, axis=0)
        # median_regret = np.append(init_regret, median_regret)

        lower_percentile = np.percentile(regret, 10, axis=0)
        # lower_percentile = np.append(init_regret, lower_percentile)

        upper_percentile = np.percentile(regret, 90, axis=0)
        # upper_percentile = np.append(init_regret, upper_percentile)
        ax[1].plot(np.arange(1, episodes + 1), mean_regret, color='k', linewidth = 5)
        ax[1].plot(np.arange(1, episodes + 1), mean_regret, label=algo_labels[algo_idx], markerfacecolor='black', marker='o', markersize=10, color=cols[algo_idx], linewidth = 3)
        # ax2.plot(range(0, episodes + 1), median_regret, label=algo, color=cols[algo_idx])
        # ax2.fill_between(np.arange(0, episodes + 1), lower_percentile, upper_percentile, alpha=0.2, color=cols[algo_idx])

    else:
        # take the log of the regret
        regret = np.log(regret)

        mean_regret = np.mean(regret, axis=0)

        # do line plots for the percentage of seeds where the regret is zero at the end of each episode
        episode_break_points = np.arange(episode_length - 1, episodes * episode_length, episode_length)

        # ax.plot(np.arange(1, episodes * episode_length + 1), mean_regret, color='black', linewidth = 5)
        # ax.plot(np.arange(1, episodes * episode_length + 1), mean_regret, label=algo_labels[algo_idx], color=cols[algo_idx], linewidth = 3)
        ax[1].plot(episode_break_points + 1, mean_regret[episode_break_points], color='black', linewidth = 5)
        ax[1].plot(episode_break_points + 1, mean_regret[episode_break_points], label=algo_labels[algo_idx], color=cols[algo_idx], linewidth = 3, markerfacecolor='black', marker='o', markersize=10)
        # ax2.plot(np.median(regret), label=algo, color=cols[algo_idx])
        # ax2.fill_between(np.arange(len(regret)), lower_percentile, upper_percentile, alpha=0.2, color=cols[algo_idx])


if episodic:
    ax[1].patch.set_visible(False)
    # do a group bar chart with the percentage of seeds where the regret is zero
    ax2 = ax[1].twinx()
    ax2.set_zorder(ax[1].get_zorder() - 1)
    # plot the group bar chart
    width = 0.2
    x = np.arange(1, episodes + 1)
    ax2.bar(x - 1.5 * width, zero_regret[0, :], width, label = algo_labels[0], color = cols[0])
    ax2.bar(x - 0.5 * width, zero_regret[1, :], width, label = algo_labels[1], color = cols[1])
    ax2.bar(x + 0.5 * width, zero_regret[2, :], width, label = algo_labels[2], color = cols[2])
    ax2.bar(x + 1.5 * width, zero_regret[3, :], width, label = algo_labels[3], color = cols[3])

    ax[1].set_xlabel('Episode', fontsize=20)
    ax[1].set_ylabel('Average Regret', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    # set y ticks at 0 and 0.035
    # calculate the maximum regret
    max_regret = np.max(mean_regret)
    # reduce to 3 decimal places
    max_regret = np.round(max_regret, 3)
    ax[1].set_yticks([0, max_regret])

    # flip the labels for the second axis
    ax2.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', which='major', labelsize=15)
    # set y ticks at 0, 0.5 and 1
    ax2.set_yticks([0, 0.5, 1])

    ax[1].legend(loc='upper right', fontsize=15)

else:
    ax[1].patch.set_visible(False)

    ax[1].set_xlabel('Iteration', fontsize=20)
    ax[1].set_ylabel('Log Regret', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    # calculate the maximum regret
    max_regret = np.max(mean_regret)
    min_regret = np.min(mean_regret)
    # reduce to 2 decimal places
    max_regret = np.round(max_regret, 0)
    min_regret = np.round(min_regret, 0)
    ax[1].set_yticks([min_regret, max_regret])
    
    ax[1].legend(loc='upper right', fontsize=15)

    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax2.patch.set_visible(False)
    ax2.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', which='major', labelsize=15)
    
    # do line plots for the percentage of seeds where the regret is zero at the end of each episode
    episode_break_points = np.arange(episode_length - 1, episodes * episode_length, episode_length)
    width = 10

    ax2.bar(episode_break_points + 1 - 1.5 * width, zero_regret[0, episode_break_points], width, label = algo_labels[0], color = cols[0])
    ax2.bar(episode_break_points + 1 - 0.5 * width, zero_regret[1, episode_break_points], width, label = algo_labels[1], color = cols[1])
    ax2.bar(episode_break_points + 1 + 0.5 * width, zero_regret[2, episode_break_points], width, label = algo_labels[2], color = cols[2])
    ax2.bar(episode_break_points + 1 + 1.5 * width, zero_regret[3, episode_break_points], width, label = algo_labels[3], color = cols[3])

plt.subplots_adjust(wspace=0.4)

# save the figure
fig.savefig(f'ypacari_joint.png', bbox_inches='tight', dpi=300)

plt.show()