import numpy as np
from mdpexplore.densities.continous_densities import NonStationaryDeltaDensity
from mdpexplore.densities.density_estimators import DeltaDensityEstimator
from mdpexplore.env.continuous_env import ContinuousEnv
from torch.nn import ModuleList
from mdpexplore.utils.nn import DeterministicModule
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from typing import Callable, Type, Union, Tuple
from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous
from mdpexplore.densities.continous_densities import ContinuousDensity
from scipy.optimize import minimize
from scipy.linalg import norm
import networkx as nx

import torch

'''
Here we include the functions that we will use to benchmark the algorithms.
'''

class Branin2D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/branin.html

        self.optimum = 1.0473939180374147
        self.dim = 2
        
        # set the hyper-parameters for this function
        self.kappa = 0.6
        self.gamma = 0.15

        self.name = 'Branin2D'

        self.beta = np.array([10, 10, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array([[2, 6.7, 8, 6, 3, 2, 5, 8, 6, 7], \
            [9, 2, 8, 6, 7, 9, 3, 1, 2, 3.6], \
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    
    def query_function(self, x):
        x1 = x[:, 0] + 0.5
        x2 = x[:, 1] + 0.5

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6)**2
        s2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar) - 44.81

        return -(s1 + s2) / 51.95

class ModifiedBranin2D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/branin.html, slightly modified to have a unique optimum

        self.optimum = 1.2943814926103765
        self.dim = 2
        
        # set the hyper-parameters for this function
        self.kappa = 0.65
        self.gamma = 0.15

        self.name = 'ModifiedBranin2D'

        self.beta = np.array([10, 10, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array([[2, 6.7, 8, 6, 3, 2, 5, 8, 6, 7], \
            [9, 2, 8, 6, 7, 9, 3, 1, 2, 3.6], \
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    
    def query_function(self, x):
        x1 = x[:, 0] + 0.5
        x2 = x[:, 1] + 0.5

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6)**2
        s2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar) - 44.81

        # now add a very small linear function of x1 to have a unique optimum
        s3 = 0.02 * x1bar
        s4 = 0.02 * x2bar

        return -(s1 + s2) / 51.95 + s3 + s4

class Hartmann3D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/hart3.html
        self.optimum = 3.8627801
        self.dim = 3

        self.kappa = 2.0
        self.gamma = 0.13849

        self.name = 'Hartmann3D'

        self.A = np.array( \
            [[3, 10, 30], \
                [0.1, 10, 35], \
                    [3, 10, 30], \
                        [0.1, 10, 35]])
        
        self.P = 1e-4 * np.array( \
            [[3689, 1170, 2673], \
                [4699, 4387, 7470], \
                    [1091, 8732, 5547], \
                        [381, 5743, 8828]])

        self.alpha = np.array([1, 1.2, 3, 3.2])
    
    def query_function(self, x):
        x = x + 0.5
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 3):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Hartmann6D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/hart6.html
        self.optimum = 3.322368011391339
        self.dim = 6

        self.kappa = 1.7
        self.gamma = 0.22

        self.name = 'Hartmann6D'

        self.A = np.array( \
            [[10, 3, 17, 3.5, 1.7, 8], \
                [0.05, 10, 17, 0.1, 8, 14], \
                    [3, 3.5, 1.7, 10, 17, 8], \
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        self.P = 1e-4 * np.array( \
            [[1312, 1696, 5569, 124, 8283, 5886], \
                [2329, 4135, 8307, 3736, 1004, 9991], \
                    [2348, 1451, 3522, 2883, 3047, 6650], \
                        [4047, 8828, 8732, 5743, 1091, 381]])

        self.alpha = np.array([1, 1.2, 3, 3.2])
    
    def query_function(self, x):
        x = x + 0.5
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 6):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Michalewicz2D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/michal.html
        self.optimum = 0.6754469275474548
        self.dim = 2

        self.kappa = 0.35
        self.gamma = 0.179485

        self.name = 'Michaelwicz2D'

        self.m = 10
    
    def query_function(self, x):
        x = x + 0.5
        x = x * np.pi
        S1 = np.sin(x[:, 0]) * (np.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = np.sin(x[:, 1]) * (np.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        return S1 + S2

class Michalewicz3D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/michal.html
        self.optimum = 1.6728426380418773
        self.dim = 3

        self.kappa = 0.85
        self.gamma = 0.179485

        self.name = 'Michaelwicz3D'

        self.m = 10
    
    def query_function(self, x):
        x = x + 0.5
        x = x * np.pi
        S1 = np.sin(x[:, 0]) * (np.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = np.sin(x[:, 1]) * (np.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        S3 = np.sin(x[:, 2]) * (np.sin(3 * x[:, 2] / np.pi))**(2*self.m)

        return S1 + S2 + S3

class Levy4D():
    def __init__(self):
        # taken from website: https://www.sfu.ca/~ssurjano/levy.html
        self.optimum = 1.0
        self.dim = 4

        self.kappa = 0.6
        self.gamma = 0.14174832

        self.name = 'Levy4D'
    
    def query_function(self, x):
        x = x * 16
        
        w = 1 + (x[:, 0] - 1) / 4
        S = np.sin(np.pi * w)**2
        for d in range(3):
            w = 1 + (x[:, d] - 1) / 4
            S += (w - 1)**2 * (1 + 10 * np.sin(np.pi * w + 1)**2)
        
        w = 1 + (x[:, 3] - 1) / 4
        S += (w - 1)**2 * (1 + np.sin(2 * np.pi * w)**2)

        return -S / 400 + 1

'''
Here we implement a dummy convex solver that will be used to implement the SnAKe algorithm.
'''


class TruncatedSnAKeSolver(ConvexSolverBase):
    def __init__(self, env : ContinuousEnv, objective, verbosity = 0, accuracy = 1e-4, epsilon = 0.1) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'Truncated SnAKe'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
        self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        self.epsilon = epsilon

        # start an array of visited states
        self.visited_states = np.zeros((0, self.env.states_dim))
    
    def optimize(self, emissions, visitations, episodes) -> None:
        '''
        Builds and optimization path using the SnAKe algorithm.
        '''
        # pre-set variables for emmissions, visitations and episodes
        self.emissions = emissions
        self.visitations = visitations
        self.episodes = episodes
        self.dim = self.env.states_dim

        # first look at the planning horizon
        H_plan = self.env.max_episode_length - self.env.h

        # calculate the current density function
        density = self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.stationary)

        # obtain array of thompson samples
        potential_maximizers = self.objective.set_of_maximizers

        # do point deletion
        points_already_queried = np.array(self.visitations[0][0])
        
        for pq in points_already_queried:
            # find the closest point in the array of potential maximizers
            distances = np.linalg.norm(pq - potential_maximizers, axis = 1)
            closest_point = np.argmin(distances)
            dist = distances[closest_point]
            # delete the point
            if dist < self.epsilon:
                potential_maximizers = np.delete(potential_maximizers, closest_point, axis = 0)
            # else delete a random point
            else:
                rand_idx = np.random.randint(0, potential_maximizers.shape[0])
                potential_maximizers = np.delete(potential_maximizers, rand_idx, axis = 0)

        # add the current state to the array of potential maximizers
        potential_maximizers = np.concatenate((self.env.state.reshape(1, -1), potential_maximizers), axis = 0)

        # build an adjacency matrix of the distances between the potential maximizers
        distances = np.zeros((potential_maximizers.shape[0], potential_maximizers.shape[0]))
        for i in range(potential_maximizers.shape[0]):
            for j in range(potential_maximizers.shape[0]):
                distances[i, j] = norm(potential_maximizers[i, :] - potential_maximizers[j, :])
        

        # build a graph from the adjacency matrix
        graph = nx.convert_matrix.from_numpy_array(distances)
        # define the algorithm to solve the TSP
        tsp = nx.algorithms.approximation.traveling_salesman_problem
        SA_tsp = nx.algorithms.approximation.simulated_annealing_tsp
        # solve the TSP
        method = lambda G, wt: SA_tsp(G, 'greedy', weight = wt, source = 0)
        # obtain the path
        new_path_idx = tsp(graph, cycle = True, method = method)
        # obtain the path
        new_path = potential_maximizers[new_path_idx, :]
        # remove the last element of the path (which is the same as the first, since it is a cycle)
        new_path = new_path[:-1, :]
        # now create array of actions, we move
        actions = np.zeros((H_plan, self.env.actions_dim))

        # we follow the path defined by the TSP with the movement constraints until we reach H_plan or the end of the path
        current_state = new_path[0, :]
        next_state = new_path[1, :]
        h_idx = 0
        path_idx = 1

        # now calculate the actions with a truncated policy
        while h_idx < H_plan:
            # loop until current state is approximately equal to next state
            # while (norm(current_state - next_state) > 1e-6) and (h_idx < H_plan):
            # check direction of movement
            direction = next_state - current_state
            # check if direction is valid
            if np.all(direction <= self.env.max_action) and np.all(direction >= self.env.min_action):
                # if it is valid, move in that direction
                actions[h_idx, :] = direction
                current_state = next_state
                next_state = new_path[np.minimum(path_idx + 1, H_plan), :]
                path_idx += 1
            else:
                # clip the direction
                direction = np.clip(direction, self.env.min_action, self.env.max_action)
                actions[h_idx, :] = direction
                current_state = current_state + direction

            # update the idx
            h_idx += 1
            

        # now create the policy
        policy = ModuleList()
        # define actions as a tensor
        actions = torch.tensor(actions, requires_grad = False)
        for h in range(self.env.max_episode_length - self.env.h):
            policy.append(DeterministicModule(actions[h, :]))

        output_policy = NonStationaryPolicyContinuous(self.env, policy)
        # create summarized policy
        self.policies.append(output_policy)
        self.weights = [1.0]
        self.summarize()

        return self.summarized_policy, self.policies, self.weights, self.densities


if __name__ == '__main__':
    func = Michalewicz2D()
    theta_star = lambda x: - func.query_function(x.reshape(1, -1))

    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    best_func_val = np.inf

    for i in range(1000):
        x0 = np.random.uniform(-0.5, 0.5, (1, 2)).reshape(-1)
        res = minimize(theta_star, x0, method = 'L-BFGS-B', bounds = bounds, tol = 1e-20, options = {'maxiter': 10000})
        print(res.x)
        print(res.fun)
        print('-----------------')
        if res.fun < best_func_val:
            best_func_val = res.fun
            best_x = res.x
    
    print('Best value found: ', best_func_val)
    print('Best point found: ', best_x)