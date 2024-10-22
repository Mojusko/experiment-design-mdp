import os
import random

import numpy as np
import torch
from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import argparse

from doexpy.solvers.additive_gradient import AdditiveGradient
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
# from doexpy.convex_solvers.cyipopt import InteriorPoint
from doexpy.convex_solvers.greedy_approximation import ContinuousGreedyApproximation
from doexpy.convex_solvers.first_action_random_path import RandomPaths
from doexpy.mdpexplore import MdpExplore
from doexpy.env.continuous_bandits import ContinuousMovementConstrainedBayesianOptimization
from doexpy.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominatorContinuous
from doexpy.feedback.bandit_feedback import ContinuousBanditFeedbackAsynchronous

from  experiments.snar.fit_snar import SnAr, LSR
from experiments.asynch_benchmarks.fit_asynch_benchmarks import TruncatedSnAKeSolver

from scipy.stats.qmc import Sobol

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='SnAr Benchmark.')
    # arguments I know I will need
    parser.add_argument('--seed', default=121, type=int, help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--episode_length', default=100, type=int, help='Length of the episode')
    parser.add_argument('--noise', default=0.001, type=float, help='Noise variance')
    parser.add_argument('--number_of_maximizers', default=100, type=int, help='Number of Thompson Samples, UCB samples, etc...')
    parser.add_argument('--maximizer_type', default='thompson_sampling', type=str, help='Type of maximizer (thompson_sampling / ucb / af_mix)')
    parser.add_argument('--delta_mov', default= 0.1, type=float, help='Maximum movement constraint')
    parser.add_argument('--num_features', default=512, type=int, help='Number of features')
    parser.add_argument('--num_components', default=1, type=int, help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=1, type=int, help='Number of episodes')
    parser.add_argument('--policy', default='density', type=str, help='Summarized policy type (mixed/average/density)')
    parser.add_argument('--snake', default=False, type=bool, help='Wether to use the snake algorithm or not')
    parser.add_argument('--random_paths', default=False, type=bool, help='Wether to use the random paths algorithm or not')
    parser.add_argument('--num_random_paths', default=100, type=int, help='Number of random paths')
    parser.add_argument('--delay', default=25, type=int, help='Delay in the feedback')
    # extra arguments
    parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
    parser.add_argument('--uncertain', default="false", type = str, help = "type")
    parser.add_argument('--random', default="false", type=str, help="type")

    args = parser.parse_args()
	
    # set default tensor type
    torch.set_default_tensor_type(torch.DoubleTensor)
    # set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define the function to optimize
    func = SnAr()

    theta_star = lambda x: func.query_function(x.reshape(-1, func.dim))
    # set noise level
    sigma = np.sqrt(args.noise)

    # set regularization parameter
    lambd = 1.0
    # maximum movement parameter
    delta_mov = args.delta_mov
    # number of maximizers
    if not args.snake:
        num_maximizers = args.number_of_maximizers
    else:
        num_maximizers = args.episode_length + 1

    # define the kernel
    # generate a finite grid of points
    grid_points = 2 ** (func.dim + 6)

    sobol_generator = Sobol(d=func.dim, scramble=True, seed=args.seed)
    action_space = sobol_generator.random(grid_points) - 0.5

    # define the embedding
    kernel_rbf = KernelFunction(kernel_name='squared_exponential', d = func.dim, kappa = func.kappa, gamma = func.gamma)
    embedding = NystromFeatures(m = torch.tensor(args.num_features), kernel_object = kernel_rbf)
    embedding.fit_gp(torch.tensor(action_space), None)

    # finally define the estimator
    estimator = KernelizedFeatures(embedding, m = args.num_features, s = sigma, lam = lambd, d = func.dim, diameter = 0.5)

    env = ContinuousMovementConstrainedBayesianOptimization(
        states_dim = func.dim,
        actions_dim = func.dim,
        theta_star=theta_star,
        sigma=sigma,
        min_action = -args.delta_mov,
        max_action = args.delta_mov,
        max_episode_length = args.episode_length)
    
    design = DesignBestArmLinearBanditNoDenominatorContinuous(
        env = env,
        lambd = lambd,
        sigma = sigma,
        embedding = embedding,
        num_of_maximizers = num_maximizers,
    )

    if args.snake:
        # define the convex solver
        convex_solver = TruncatedSnAKeSolver(env, objective=design, epsilon = func.gamma)
    elif args.random_paths:
        convex_solver = RandomPaths(env, objective=design, num_paths = args.num_random_paths)
    else:
        convex_solver = ContinuousGreedyApproximation(env, objective=design)
    
    # initial point
    init_point = np.ones((1, func.dim)) * -0.5

    # define the feedback class
    if args.maximizer_type == 'ucb':
        maximization_set_method = 'ucb'
    elif args.maximizer_type == 'af_mix':
        maximization_set_method = 'af_mix'
    else:
        maximization_set_method = 'thompson_sampling'

    feedback = ContinuousBanditFeedbackAsynchronous(env, 
                                                    design, 
                                                    estimator, 
                                                    theta_star = theta_star, 
                                                    sigma = sigma, 
                                                    video = False, 
                                                    markovian = False, 
                                                    maximization_set_method = maximization_set_method,
                                                    initial_point = init_point,
                                                    keep_track_best_guess = True,
                                                    asynchronous_delay= args.delay)

    # define the MDP explore algorithm
    me = MdpExplore(
        env=env,
        objective=design,
        convex_solver=convex_solver,
        verbosity=args.verbosity,
        feedback=feedback,
        general_policy = 'non-markovian'
    )

    # run the experiment
    val, opt_val, visitations = me.run(
        episodes=args.episodes,
        return_visitations = True
    )

    # save the results
    x_evaluations = np.zeros((args.episodes, args.episode_length + 1, func.dim))
    for ep in range(args.episodes):
        x_evaluations[ep, :] = np.array(visitations[ep][0]).reshape(args.episode_length + 1, -1)
    
    x_evaluations = x_evaluations.reshape(args.episodes * (args.episode_length + 1), func.dim)

    f_evaluations = []
    for x in x_evaluations:
        f_evaluations.append(theta_star(x.reshape(1, -1)))
    
    f_evaluations = np.array(f_evaluations).reshape(-1, 1)

    best_guesses = np.array(feedback.best_arm)

    file_name = f'experiments/snar/results/asynch' + func.name + f'/delay_{args.delay}/delta_mov_{args.delta_mov}/noise_var_{args.noise}/num_features_{args.num_features}/episode_length_{args.episode_length}/'
    
    if args.snake:
        algo_name = '/TruncatedSnAKe'
    elif args.random_paths:
        algo_name = f'/MDPExploreRandomPaths/num_paths_{args.num_random_paths}'
    else:
        algo_name = f'/MDPExplore/' + maximization_set_method + f'/num_maximizers_{num_maximizers}'

    file_name = file_name + algo_name + f'/seed_{args.seed}/'

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    np.save(file_name + 'x_evaluations.npy', x_evaluations)
    np.save(file_name + 'f_evaluations.npy', f_evaluations)
    np.save(file_name + 'best_guesses.npy', best_guesses)