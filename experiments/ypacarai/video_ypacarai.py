import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from PIL import Image

from experiments.ypacarai.fit_ypacarai import Schekel2D

# plot either bandit or EI
plot = 'bandit'
episode_length = 50
episodes = 3

# load the lake image in grayscale
lake = Image.open('ypacarai_lake.jpg').convert('L')
lake = np.asarray(lake)
# make pixel values either 0 or 1
lake = (lake < 128).astype(int)

# define the function to optimize
scheckel_function = Schekel2D()
theta_star = lambda x: scheckel_function.query_function(x.reshape(-1, 2))

# load the action space
action_space = np.load('ypacarai_centroids.npy')
optimal_action = action_space[np.argmax(theta_star(action_space))]

# initial and terminal state
init_state = 59
terminal_state = 59

# load the ucb, lcb, mean and actions
ucbs = np.load('ucb.npy')
lcbs = np.load('lcb.npy')
means = np.load('mean.npy')
actions = np.load('actions.npy')

# initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# set the axis limits
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)

# make contour plot of the true function
x = np.linspace(-0.6, 0.6, 101)
y = np.linspace(-0.6, 0.6, 101)
X, Y = np.meshgrid(x, y)
Z = np.zeros((101, 101))

for i in range(100):
    for j in range(100):
        f = theta_star(np.array([X[i, j], Y[i, j]]))
        Z[i, j] = f

# plot the contour
contour_plot = ax.contourf(X, Y, Z, 20, cmap='Blues')
# add a colorbar
fig.colorbar(contour_plot, ax=ax)

# initialize the iteration text
iteration_text_h = ax.text(0.42, 0.35, '', ha = 'right', va = 'bottom', fontsize = 10, zorder = 20)
iteration_text_ep = ax.text(0.4, 0.4, '', ha = 'right', va = 'bottom', fontsize = 10, zorder = 20)
# scatter plot of discarded actions, with a circle marker
discarded_region_plot = ax.scatter([], [], color='red', marker='o', label='Discarded actions', s = 30)
# scatter plot of non-discarded actions
non_discarded_region_plot = ax.scatter([], [], color='darkorange', marker='o', label='Non-discarded actions', s = 30)
# plot the action taken with a scatter, using a black triangle
action_taken = ax.scatter([], [], color='black', marker='2', s=250)
# plot the best action with a scatter, using a yellow star
best_action = np.argmax(theta_star(action_space))
best_action = ax.scatter(action_space[best_action, 0], action_space[best_action, 1], color='yellow', marker='*', label='Best action', s=150)
# plot the initial state with a diamond
initial_state = ax.scatter(action_space[init_state, 0], action_space[init_state, 1], color='black', marker='s', label='Initial / Terminal state', s=100, zorder=11)
# plot the terminal state with a square
terminal_state = ax.scatter(action_space[terminal_state, 0], action_space[terminal_state, 1], color='black', marker='s', s=100, zorder=11)
# initialize the path followed
path = ax.plot([], [], color='black', label='Path followed', linewidth=2, linestyle='--')

# invert the lake for colourscheme
lake = 1 - lake

# finally plot the lake, but masking the values where the lake is 0
lake = np.ma.masked_where(lake == 0, lake)
# create a color map based on Greens that only gets to half the intensity
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkseagreen", "darkseagreen"])
ax.imshow(lake, cmap=cmap, extent=[-0.5, 0.5, -0.5, 0.5], zorder=10)
# make the lake background green
ax.set_facecolor('darkseagreen')

# label the axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')

title = 'Constrained Ypacarai Example'

ax.set_title(title)

# add a legend lower left
leg = ax.legend(loc='lower left')
leg.set_zorder(20)

def frames():
    # load the arrays from memory
    actions = np.load('actions.npy')
    lcbs = np.load('lcb.npy')
    means = np.load('mean.npy')
    ucbs = np.load('ucb.npy')

    for i in range(len(actions) - 1):
        for j, a in enumerate(actions[i+1]):
            yield actions[:i+1], lcbs[i, :], means[i, :], ucbs[i, :], (i, j)

    for i in range(5):
        yield actions, lcbs[-1, :], means[-1, :], ucbs[-1, :], (len(actions) - 1, 0)

# define animation function
def animate(params):
    action = int(params[0][-1])
    action_list = params[0]
    lcb = params[1]
    mean = params[2]
    ucb = params[3]
    idx_episode, idx_h = params[4]

    idx_ep = idx_episode // episode_length + 1
    idx_step = idx_episode % episode_length + 1

    # calculate the discarded region
    discarded_region = ucb < np.max(lcb)
    non_discarded_region = ucb >= np.max(lcb)

    # append the initial state to the action list
    action_list = np.append(np.array(init_state), action_list)

    # update the action taken
    action_taken.set_offsets(np.array(action_space[action]).reshape(-1, 2))
    # update the discarded region
    discarded_region_plot.set_offsets(np.array(action_space[discarded_region]))
    # the non-discarded region
    non_discarded_region_plot.set_offsets(np.array(action_space[non_discarded_region]))
    # update the path
    lower_splice = (idx_ep - 1) * episode_length
    upper_splice = idx_ep * episode_length + idx_step
    action_list = action_list[lower_splice:upper_splice]
    # append the initial state to the action list
    # action_list = np.append(np.array(init_state), action_list)
    path[0].set_data(action_space[action_list.astype(int), 0], action_space[action_list.astype(int), 1])

    # set the text
    iteration_text_h.set_text("Steps taken: " + str(idx_step) + f' / {episode_length}')
    iteration_text_ep.set_text("Episode: " + str(idx_ep) + f' / {episodes}')

def init_func():
    pass

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
ani = FuncAnimation(fig=fig, func=animate, frames=frames, repeat=True, save_count=300, interval=1000, init_func=init_func)
writer = animation.PillowWriter(fps=1)


# save name
if plot == 'bandit':
    save_name = 'constrained_ypacarai_bandit'
elif plot == 'EI':
    save_name = 'constrained_ypacarai_EI'

save_name = save_name + '.gif'

ani.save(save_name, writer=writer)