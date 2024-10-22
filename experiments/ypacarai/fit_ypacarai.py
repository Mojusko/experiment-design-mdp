import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from doexpy.env.bandits import MovementConstrainedBayesianOptimization

'''
In this script we define the Ypacarai function to be solved. As in previous works, we base it off the Scheckel function.
'''

class Schekel2D():
    def __init__(self, n_optims = 2):
        # taken from website: https://www.sfu.ca/~ssurjano/shekel.html

        self.optimum = -11
        self.dim = 2

        self.name = 'Schekel2D'

        self.num_of_optims = n_optims
        self.beta = np.array([10, 10, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array([[2, 6.7, 8, 6, 3, 2, 5, 8, 6, 7], \
            [9, 2, 8, 6, 7, 9, 3, 1, 2, 3.6], \
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    
    def query_function(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = (x + 0.5) * 10
        S1 = 0
        for i in range(self.num_of_optims):
            S2 = 0
            for j in range(2):
                S2 = S2 + (x[:, j] - self.C[i, j])**2
            S1 = S1 + 1 / (S2 + self.beta[i])
        return 10 * S1

class YpacaraiEnv(MovementConstrainedBayesianOptimization):
    def __init__(self,
                 adjacency_matrix: np.array,
                 action_space: np.array, 
                 action_space_pre_embedding: np.array, 
                 theta_star: np.array, sigma: float, 
                 discount_factor: float = 0.99, 
                 max_episode_length: int = 10,
                 init_state: int = 0,
                 terminal_state: int = None) -> None:
        
        super().__init__(action_space, action_space_pre_embedding, theta_star, sigma, discount_factor, max_episode_length, init_state)
        self.adjacency_matrix = adjacency_matrix
        if terminal_state is None:
            self.constrained = False
        else:
            self.constrained = True
            self.terminal_state = terminal_state
    
    def is_valid_action(self, action, state) -> bool:
        return self.adjacency_matrix[state, action] == 1

if __name__ == '__main__':
    # get the ypacarai function action space
    ypacarai_action_space = np.load('ypacarai_centroids.npy')

    # obtain the function values at the action space
    ypacarai_function = Schekel2D()
    ypacarai_function_values = ypacarai_function.query_function(ypacarai_action_space)

    # load the lake image
    lake = Image.open('ypacarai_lake.jpg').convert('L')

    # plot the action space and the function values with the corresponding colors
    # initialize a plot
    fig = plt.figure(figsize=(8, 6))

    # plot the lake as a background
    plt.imshow(lake, cmap='Greens', extent=[-0.5, 0.5, -0.5, 0.5])

    # plot the actions
    plt.scatter(ypacarai_action_space[:, 0], ypacarai_action_space[:, 1], c=ypacarai_function_values, s=10)

    # plot a star near the optimum action
    optim = ypacarai_action_space[np.argmax(ypacarai_function_values)]
    plt.scatter(optim[0], optim[1], marker='*', s=100, c='r')

    plt.show()

    print('hola')