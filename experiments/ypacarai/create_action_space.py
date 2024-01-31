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


'''
We can create a smaller action space of 100 points using k-means clustering, do it once and save it
'''
# kmeans = KMeans(n_clusters=100, random_state=0).fit(action_space)
# ypacarai_action_space = kmeans.cluster_centers_
# save the action space
# np.save('ypacarai_centroids.npy', ypacarai_action_space)
# load the action space
ypacarai_action_space = np.load('ypacarai_centroids.npy')

'''
We can create the available actions space by using an adjacency matrix and correcting it manually, once done save it
'''

# ypacarai_adjacency_matrix = np.zeros((ypacarai_action_space.shape[0], ypacarai_action_space.shape[0]))

# # action allowed if the distance is less than 0.1
# for i in range(ypacarai_action_space.shape[0]):
#     for j in range(ypacarai_action_space.shape[0]):
#         if np.linalg.norm(ypacarai_action_space[i] - ypacarai_action_space[j]) < 0.1:
#             ypacarai_adjacency_matrix[i, j] = 1

# # remove the following connections
# # 59 to 22, 21 to 5, 58 to 5, 58 to 88, 99 to 88, 99 to 44, 16 to 81, 2 to 43, 16 to 44, 45 to 65, 18 to 43
# ypacarai_adjacency_matrix[59, 22] = 0
# ypacarai_adjacency_matrix[22, 59] = 0
# ypacarai_adjacency_matrix[21, 5] = 0
# ypacarai_adjacency_matrix[5, 21] = 0
# ypacarai_adjacency_matrix[58, 5] = 0
# ypacarai_adjacency_matrix[5, 58] = 0
# ypacarai_adjacency_matrix[58, 88] = 0
# ypacarai_adjacency_matrix[88, 58] = 0
# ypacarai_adjacency_matrix[99, 88] = 0
# ypacarai_adjacency_matrix[88, 99] = 0
# ypacarai_adjacency_matrix[99, 44] = 0
# ypacarai_adjacency_matrix[44, 99] = 0
# ypacarai_adjacency_matrix[16, 81] = 0
# ypacarai_adjacency_matrix[81, 16] = 0
# ypacarai_adjacency_matrix[2, 43] = 0
# ypacarai_adjacency_matrix[43, 2] = 0
# ypacarai_adjacency_matrix[16, 44] = 0
# ypacarai_adjacency_matrix[44, 16] = 0
# ypacarai_adjacency_matrix[45, 65] = 0
# ypacarai_adjacency_matrix[65, 45] = 0
# ypacarai_adjacency_matrix[18, 43] = 0
# ypacarai_adjacency_matrix[43, 18] = 0

# # save the adjacency matrix
# np.save('ypacarai_adjacency_matrix.npy', ypacarai_adjacency_matrix)

# load the adjacency matrix
ypacarai_adjacency_matrix = np.load('ypacarai_adjacency_matrix.npy')

'''
We create a plot to see the action space and the adjacency matrix are correct. Plot to be used in the paper.
'''

# initialize a plot
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
# make the background green
ax.set_facecolor('darkseagreen')

# plot the lake first as a scatter plot
ax.scatter(lake_scatter[:, 0], lake_scatter[:, 1], s=0.1)

# plot the centroids dark orange
ax.scatter(ypacarai_action_space[:, 0], ypacarai_action_space[:, 1], s=30, c='darkorange')

# plot the index of the centroids to see which one transitions to manually remove
# for i in range(ypacarai_action_space.shape[0]):
#    ax.text(ypacarai_action_space[i, 0], ypacarai_action_space[i, 1], str(i), fontsize=10, color='white')

# define the initial state
init_state = 59
# define the terminal state
terminal_state = 43
optimal_state = 95
# plot a diamond near the initial state and a square near the terminal state
plt.scatter(ypacarai_action_space[init_state, 0], ypacarai_action_space[init_state, 1], marker='s', s=150, c='k', label='initial state')
# plt.scatter(ypacarai_action_space[terminal_state, 0], ypacarai_action_space[terminal_state, 1], marker='s', s=100, c='k', label='terminal state')
plt.scatter(ypacarai_action_space[95, 0], ypacarai_action_space[95, 1], marker='*', s=150, c='darkorange', label='initial state', zorder = 10)
plt.scatter(ypacarai_action_space[95, 0], ypacarai_action_space[95, 1], marker='*', s=500, c='k', label='initial state', zorder = 9)

plt.scatter(ypacarai_action_space[53, 0], ypacarai_action_space[53, 1], marker='s', s=100, c='darkorange', label='initial state', zorder = 10)
plt.scatter(ypacarai_action_space[53, 0], ypacarai_action_space[53, 1], marker='s', s=200, c='k', label='initial state', zorder = 9)

# create a line between the centroids if they are connected
for i in range(ypacarai_action_space.shape[0]):
    for j in range(ypacarai_action_space.shape[0]):
        if ypacarai_adjacency_matrix[i, j] == 1:
            ax.plot([ypacarai_action_space[i, 0], ypacarai_action_space[j, 0]], [ypacarai_action_space[i, 1], ypacarai_action_space[j, 1]], c = 'black')

# save the figure as a png
plt.savefig('ypacarai_constrained.png', dpi=300)

plt.show()