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

# initialize the figure
fig, ax = plt.subplots(figsize=(6, 6))

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

leg = ax.legend(framealpha = 1, fontsize = 14)
leg.set_zorder(20)

# set limits
ax.set_xlim([-0.55, 0.55])
ax.set_ylim([-0.55, 0.55])

# set axis labels
ax.set_xlabel(r'residence time ($\tau$)', fontsize = 14)
ax.set_ylabel('reactant stoichiometry (B)', fontsize = 14)

positions = [-0.5, -0.25, 0, 0.25, 0.5]
labels1 = [0, 10, 20, 30, 40]
labels2 = [0.0, 0.25, 0.5, 0.75, 1.0]

ax.set_xticks(positions, labels1, fontsize = 14)
ax.set_yticks(positions, labels2, fontsize = 14)

# save the figure
plt.savefig('flow_ode_mono.png', dpi = 300, bbox_inches = 'tight')

plt.show()