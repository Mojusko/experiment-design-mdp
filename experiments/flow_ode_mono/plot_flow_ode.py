import matplotlib.pyplot as plt
import numpy as np

from experiments.flow_ode.fit_flow_ode import SchreckerODE

# define the function to optimize
ode_solution = SchreckerODE()
theta_star = lambda x: ode_solution.solve(t_span = np.linspace(0, (x[:, 0].item() + 0.5) * 40), y0 = [0, 1 - (x[:, 1].item() + 0.5), x[:, 1].item() + 0.5, 0, 0, 0, 0])[:, 0][-1]

# define the action space
grid_1d = np.linspace(-0.5, 0.5, 11)
grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
action_space = grid

# contour plot of theta_star
x = np.linspace(-0.5, 0.55, 101)
y = np.linspace(-0.5, 0.5, 101)
X, Y = np.meshgrid(x, y)
Z = np.zeros((101, 101))

for i in range(100):
    for j in range(100):
        f = theta_star(np.array([X[i, j], Y[i, j]]).reshape(-1, 2))
        Z[i, j] = f

# load the paths taken by the algorithm
paths = np.load('flow_mono_paths.npy')
ucbs = np.load('flow_mono_ucbs.npy')
lcbs = np.load('flow_mono_lcbs.npy')

# initialize the figure with two subplots
# fig, ax = plt.subplots(figsize=(6, 6))
import matplotlib.gridspec as gridspec
# Create a figure with a specific overall size
fig = plt.figure(figsize=(16, 5))  # Total figure size: length=14, height=6
gs = gridspec.GridSpec(1, 2, width_ratios=[8, 8])

# Add the first subplot (length 6, height 6)
ax = fig.add_subplot(gs[1])  # 1 row, 2 columns, 1st subplot
# ax.set_title('Subplot 1')

# Add the second subplot (length 8, height 6)
ax2 = fig.add_subplot(gs[0])  # 1 row, 2 columns, 2nd subplot


# plot the contour
contour_plot = ax.contourf(X, Y, Z, 20, cmap='Blues')

# plot the paths
for path in reversed(paths):
    ax.plot(path[:, 0], path[:, 1], alpha = 0.75, linewidth = 5)

max_point = action_space[np.argmax([theta_star(act.reshape(-1, 2)) for act in action_space])].reshape(1, -1)
# plot the true maximizer
ax.scatter(max_point[:, 0], max_point[:, 1], marker='*', color='darkorange', label='True maximizer', zorder = 11, s = 150)
ax.scatter(max_point[:, 0], max_point[:, 1], marker='*', color='k', zorder = 10, s = 500)

# find the set of potential maximizers
mask = ucbs > np.max(lcbs)
mask[114] = False
potential_maximizers = action_space[mask]
ax.scatter(potential_maximizers[:, 0], potential_maximizers[:, 1], marker='o', color='darkorange', label='Potential maximizers', zorder = 10, s = 50)
ax.scatter(potential_maximizers[:, 0], potential_maximizers[:, 1], marker='o', color='k', zorder = 9, s = 125)

# now plot the discarded points
mask = np.logical_not(mask)
mask[114] = False
discarded_maximizers = action_space[mask]
ax.scatter(discarded_maximizers[:, 0], discarded_maximizers[:, 1], marker='o', color='darkblue', label='Discarded states', zorder = 10)

leg = ax.legend(framealpha = 1, fontsize = 16)
leg.set_zorder(20)

# set limits
ax.set_xlim([-0.55, 0.55])
ax.set_ylim([-0.55, 0.55])

# set axis labels
ax.set_xlabel(r'residence time ($\tau$)', fontsize = 20)
ax.set_ylabel('ratio of initial reactants (B)', fontsize = 20)

positions = [-0.5, -0.25, 0, 0.25, 0.5]
labels1 = [0, 10, 20, 30, 40]
labels2 = [0.0, 0.25, 0.5, 0.75, 1.0]

ax.set_xticks(positions, labels1, fontsize = 20)
ax.set_yticks(positions, labels2, fontsize = 20)

# now create the regret plot
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
        ax2.plot(np.arange(1, episodes + 1), mean_regret, color='k', linewidth = 5)
        ax2.plot(np.arange(1, episodes + 1), mean_regret, label=algo_labels[algo_idx], markerfacecolor='black', marker='o', markersize=10, color=cols[algo_idx], linewidth = 3)
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
        ax2.plot(episode_break_points + 1, mean_regret[episode_break_points], color='black', linewidth = 5)
        ax2.plot(episode_break_points + 1, mean_regret[episode_break_points], label=algo_labels[algo_idx], color=cols[algo_idx], linewidth = 3, markerfacecolor='black', marker='o', markersize=10)
        # ax2.plot(np.median(regret), label=algo, color=cols[algo_idx])
        # ax2.fill_between(np.arange(len(regret)), lower_percentile, upper_percentile, alpha=0.2, color=cols[algo_idx])


if episodic:
    ax2.patch.set_visible(False)
    # do a group bar chart with the percentage of seeds where the regret is zero
    ax22 = ax2.twinx()
    ax22.set_zorder(ax.get_zorder() - 1)
    # plot the group bar chart
    width = 0.2
    x = np.arange(1, episodes + 1)
    ax22.bar(x - 1.5 * width, zero_regret[0, :], width, label = algo_labels[0], color = cols[0])
    ax22.bar(x - 0.5 * width, zero_regret[1, :], width, label = algo_labels[1], color = cols[1])
    ax22.bar(x + 0.5 * width, zero_regret[2, :], width, label = algo_labels[2], color = cols[2])
    ax22.bar(x + 1.5 * width, zero_regret[3, :], width, label = algo_labels[3], color = cols[3])

    ax2.set_xlabel('Episode', fontsize=20)
    ax2.set_ylabel('Average Regret', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # set y ticks at 0 and 0.035
    # calculate the maximum regret
    # max_regret = np.max(mean_regret)
    # reduce to 3 decimal places
    max_regret = np.round(max_regret, 3)
    ax2.set_yticks([0, max_regret])
    ax2.set_xticks(np.arange(1, episodes + 1))

    # flip the labels for the second axis
    ax22.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=20, fontsize=20)
    ax22.set_ylim([0, 1])
    ax22.tick_params(axis='y', which='major', labelsize=20)
    # set y ticks at 0, 0.5 and 1
    ax22.set_yticks([0, 0.5, 1])

    ax2.legend(loc='upper right', fontsize=16)

else:
    ax2.patch.set_visible(False)

    ax2.set_xlabel('Iteration', fontsize=20)
    ax2.set_ylabel('Average Log Regret', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # calculate the maximum regret
    # reduce to 2 decimal places
    max_regret = np.round(max_regret, 0)
    min_regret = np.round(min_regret, 0)
    ax2.set_yticks([min_regret, max_regret])
    ax2.set_xticks(np.arange(1, episodes + 1) * 10)
    
    ax2.legend(loc='upper right', fontsize=16)

    ax22 = ax.twinx()
    ax22.set_zorder(ax.get_zorder() - 1)
    ax22.patch.set_visible(False)
    ax22.set_ylabel('Best Arm Identified (%)', rotation=270, labelpad=14, fontsize=20)
    ax22.set_ylim([0, 1])
    ax22.tick_params(axis='y', which='major', labelsize=20)
    
    # do line plots for the percentage of seeds where the regret is zero at the end of each episode
    episode_break_points = np.arange(episode_length - 1, episodes * episode_length, episode_length)
    width = 1

    # ax2.bar(episode_break_points + 1 - 0.5 * width, zero_regret[0, episode_break_points], width, label = algo_labels[0], color = cols[0])
    # ax2.bar(episode_break_points + 1 + 0.5 * width, zero_regret[1, episode_break_points], width, label = algo_labels[1], color = cols[1])
    ax22.bar(episode_break_points + 1 - 1.5 * width, zero_regret[0, episode_break_points], width, label = algo_labels[0], color = cols[0])
    ax22.bar(episode_break_points + 1 - 0.5 * width, zero_regret[1, episode_break_points], width, label = algo_labels[1], color = cols[1])
    ax22.bar(episode_break_points + 1 + 0.5 * width, zero_regret[2, episode_break_points], width, label = algo_labels[2], color = cols[2])
    ax22.bar(episode_break_points + 1 + 1.5 * width, zero_regret[3, episode_break_points], width, label = algo_labels[3], color = cols[3])


# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(wspace=0.5) 

# save the figure
plt.savefig('flow_ode_mono_joint.png', dpi = 300, bbox_inches = 'tight')

plt.show()