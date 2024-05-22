import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
This is the file used to create the image in the paper, as well as the action space and adjacency matrix for the 
constrained Ypacarai Lake experiment.
'''

'''
We can obtain the shape of the lake by reading the image and converting it to grayscale
'''
# read the image and convert it to grayscale using Image package
lake = Image.open('abstract_lake.png').convert('L')

# convert the image to a numpy array
lake = np.asarray(lake)

# turn into a binary image
lake = (lake > 128).astype(int)
# rotate the image three times
lake = np.rot90(lake, 3)

# define the grid of actions
grid1 = np.linspace(-0.5, 0.5, lake.shape[0])
grid2 = np.linspace(-0.5, 0.5, lake.shape[1])
grid = np.array(np.meshgrid(grid1, grid2)).T.reshape(-1, 2)

# obtain the action space
lake_scatter = grid[lake.flatten() == 1]


'''
We can create a smaller action space of 100 points using k-means clustering, do it once and save it
'''
# kmeans = KMeans(n_clusters=50, random_state=0).fit(lake_scatter)
# lake_action_space = kmeans.cluster_centers_
# save the action space
# np.save('abstract_lake_centroids.npy', lake_action_space)
# load the action space
lake_action_space = np.load('abstract_lake_centroids.npy')
# add [-0.5, 0.5] and [0.5, -0.5] to the action space
lake_action_space = np.vstack([lake_action_space, np.array([[-0.5, 0.5], [0.5, -0.5]])])

# remap from [-0.5, 0.5] x [-0.5, 0.5] to [-1, 1] x [-0.5, 0.5]
# i.e. make it so that the x-axis is [-1, 1] and the y-axis is [-0.5, 0.5]
lake_action_space[:, 0] = 1.5 * lake_action_space[:, 0]
# do the same for the lake scatter
lake_scatter[:, 0] = 1.5 * lake_scatter[:, 0]

# plot the lake
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(lake_scatter[:, 0], lake_scatter[:, 1], s=0.1)
ax.set_aspect('equal')

# set limits
plt.xlim([-0.5 * 1.5, 0.5 * 1.5])
plt.ylim([-0.5, 0.5])

# set the lake background
# make the background green
ax.set_facecolor('darkseagreen')

pic_num = 3

if pic_num == 1:
    # plot the action space
    ax.scatter(lake_action_space[:, 0], lake_action_space[:, 1], s=75, c='k')

    # create adjacency matrix
    adjacency_matrix = np.zeros((lake_action_space.shape[0], lake_action_space.shape[0]))
    for i, act1 in enumerate(lake_action_space):
        for j, act2 in enumerate(lake_action_space):
            if (np.linalg.norm(act1[0] - act2[0]) < 0.2) and (np.linalg.norm(act1[1] - act2[1]) < 0.15) and (act1[0] < act2[0] + 0.05) and (0.05 + act1[1] > act2[1]):
                adjacency_matrix[i, j] = 1

    adjacency_matrix[16, 37] = 1
    adjacency_matrix[37, 14] = 1
    adjacency_matrix[0, 42] = 1
    adjacency_matrix[49, 33] = 1
    adjacency_matrix[33, 3] = 1
    adjacency_matrix[3, 17] = 1
    adjacency_matrix[26, 10] = 1
    adjacency_matrix[10, 43] = 1
    adjacency_matrix[22, 30] = 1
    adjacency_matrix[30, 36] = 1
    adjacency_matrix[5, 9] = 1
    adjacency_matrix[10, 33] = 1
    adjacency_matrix[9, 23] = 1
    adjacency_matrix[35, 23] = 1
    adjacency_matrix[2, 18] = 1
    adjacency_matrix[18, 38] = 1
    adjacency_matrix[31, 48] = 1

    adjacency_matrix[6, 10] = 0
    adjacency_matrix[5, 23] = 0
    adjacency_matrix[13, 4] = 0
    adjacency_matrix[33, 10] = 0
    adjacency_matrix[9, -1] = 0
    adjacency_matrix[32, 49] = 0

    # plot the connections as arrows
    for i, act1 in enumerate(lake_action_space):
        for j, act2 in enumerate(lake_action_space):
            if adjacency_matrix[i, j] == 1:
                ax.arrow(act1[0], act1[1], (act2[0] - act1[0]) * 0.5, (act2[1] - act1[1]) * 0.5, head_width=0.02, head_length=0.02, fc='k', ec='k')

    # plot the node number in the action space
    # for i, act in enumerate(lake_action_space):
    #    ax.text(act[0] + 0.05, act[1], str(i), fontsize=8, color='black')
    
    # plot a big star at index 3
    ax.scatter(lake_action_space[3, 0], lake_action_space[3, 1], s=1000, c='k', marker='*')
    ax.scatter(lake_action_space[3, 0], lake_action_space[3, 1], s=250, c='orange', marker='*')

    # plot the number of plot in top left corner, with a white background
    ax.text(0.39 * 1.5 + 0.05, 0.43, '(a)', fontsize=20, color='black', bbox=dict(facecolor='white', alpha=0.9))

elif pic_num > 1:
    # plot the action space
    ax.scatter(lake_action_space[:, 0], lake_action_space[:, 1], s=25, c='k')

    potential_maximizers = [11, 30, 36, 26, 33, 3, 17, 45, 48, 0, 42, 4, 34, 35, 47]
    intensity = [1, 1, 1, 5, 3, 3, 5, 1, 1, 1, 1, 3, 3, 1, 1]

    # plot each of the potential maximizers with size given by the intensity
    for i, act in enumerate(potential_maximizers):
        ax.scatter(lake_action_space[act, 0], lake_action_space[act, 1], s=150 * intensity[i], c='k')
        ax.scatter(lake_action_space[act, 0], lake_action_space[act, 1], s=100 * intensity[i], c='orange')

    # plot the node number in the action space
    # for i, act in enumerate(lake_action_space):
    #     ax.text(act[0], act[1], str(i), fontsize=8, color='black')

    if pic_num == 2:
        # plot the number of plot in top left corner, with a white background
        ax.text(0.39 * 1.5 + 0.05, 0.43, '(b)', fontsize=20, color='black', bbox=dict(facecolor='white', alpha=0.9))

    if pic_num == 3:
        # plot the path
        path_idx = [-2, 16, 28, 11, 36, 26, 10, 33, 3, 17, 39, 13, 29, 4, 34, 35, 23, -1]
        path = lake_action_space[path_idx]
        ax.plot(path[:, 0], path[:, 1], c='k', linewidth=5)
        ax.plot(path[:, 0], path[:, 1], c='red', linewidth=3)

        # plot less likely paths
        path2_idx = [-2, 16, 2, 18, 38, 15, 31, 48, 0, 42, 27, 1, 24, 5, 9, 23, -1]
        path2 = lake_action_space[path2_idx]
        # ax.plot(path2[:, 0], path2[:, 1], c='k', linewidth=5)
        ax.plot(path2[:, 0], path2[:, 1], c='red', linewidth=3, alpha=0.5)

        path3_idx = [-2, 50, 16, 2, 44, 30, 36, 26, 45, 12, 40, 20, 47, 35, 23, -1]
        path3 = lake_action_space[path3_idx]
        # ax.plot(path3[:, 0], path3[:, 1], c='k', linewidth=5)
        ax.plot(path3[:, 0], path3[:, 1], c='red', linewidth=3, alpha=0.25)

        # plot the number of plot in top left corner, with a white background
        ax.text(0.39 * 1.5 + 0.05, 0.43, '(c)', fontsize=20, color='black', bbox=dict(facecolor='white', alpha=0.9))

# remove the x and y ticks
ax.set_xticks([])
ax.set_yticks([])

# save the figure as a pdf
plt.savefig(f'abstract_lake_{pic_num}.png', bbox_inches='tight', dpi=300)

plt.show()