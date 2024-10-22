from enum import Enum
import random

import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import argparse
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from doexpy.solvers.ddpg import DDPG
from doexpy.solvers.additive_gradient import AdditiveGradient
from doexpy.solvers.dp import DP
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.convex_solvers.cyipopt import InteriorPoint
from doexpy.mdpexplore import MdpExplore
from doexpy.env.bandits import Bandits, Bandits_Left_Right, ConstrainedMaxMovement
from doexpy.env.continuous_bandits import ContinuousMovementConstrainedBayesianOptimization
from doexpy.functionals.bandit_functionals import DesignRewardBandit, DesignBestArmLinearBandit, DesignBestArmLinearBanditNoDenominator, DesignBestArmLinearBanditNoDenominatorContinuous
from doexpy.functionals.doe_adaptive_functionals import AdaptiveDesignD
from doexpy.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy
from doexpy.feedback.bandit_feedback import BanditFeedback, ContinuousBanditFeedback

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

from scipy.stats.qmc import Sobol
from scipy.integrate import odeint
import time

from experiments.flow_ode.fit_flow_ode import ode_kernel, SchreckerODE, ode_embedding

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='Schrecker ODE Problem.')
    # arguments I know I will need
    parser.add_argument('--seed', default=121, type=int, help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--episodic_feedback', default=False, type=bool, help='Wether we use episodic feedback or not')
    parser.add_argument('--episode_length', default=50, type=int, help='Length of the episode')
    parser.add_argument('--env_type', default='discrete', type=str, help='Environment type (discrete/continuous)')
    parser.add_argument('--noise', default=0.0001, type=float, help='Noise variance')
    parser.add_argument('--number_of_maximizers', default=25, type=int, help='Number of Thompson Samples')
    parser.add_argument('--delta_mov', default=0.11, type=float, help='Maximum movement constraint')
    parser.add_argument('--num_features', default=121, type=int, help='Number of features')
    parser.add_argument('--mix_objective', default=False, type=bool, help='Wether we mix our DoE design with UCB')
    parser.add_argument('--num_components', default=1, type=int, help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=1, type=int, help='Number of episodes')
    parser.add_argument('--policy', default='density', type=str, help='Summarized policy type (mixed/average/density)')
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
    ode_solution = SchreckerODE()
    theta_star = lambda x: ode_solution.solve(t_span = np.linspace(0, (x[:, 0].item() + 0.5) * 40), y0 = [0, 1 - (x[:, 1].item() + 0.5), x[:, 1].item() + 0.5, 0, 0, 0, 0])[:, 0][-1]
    # set noise level
    sigma = np.sqrt(args.noise)


    # set regularization parameter
    lambd = 1.0
    # maximum movement parameter
    delta_mov = args.delta_mov
    # number of maximizers
    num_maximizers = 25

    # define the kernel
    # generate a 2 by 2 uniform grid of points, then embed them in the 2d space
    grid_1d = np.linspace(-0.5, 0.5, 11)
    grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
    action_space = grid

    # define a nystrom grid which has a lot of points
    # grid_1d = np.linspace(-0.5, 0.5, 201)
    # grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
    # nystrom_grid = grid

    # define the ode embedding
    embedding = ode_embedding(num_features = args.num_features, action_space = action_space, alpha_ode = 1.0, alpha_rbf = 0.001)

    # define the embedded action space for the discrete case
    embedded_action_space = embedding.embed(torch.tensor(action_space)).detach().numpy()

    # finally define the estimator
    estimator = KernelizedFeatures(embedding, m = args.num_features, s = sigma, lam = lambd, d = 2, diameter = 0.5)

    # define the environment and reward function
    if args.env_type == 'continuous':
        env = ContinuousMovementConstrainedBayesianOptimization(
            states_dim = 2,
            actions_dim = 2,
            theta_star=theta_star,
            sigma=sigma,
            min_action = -delta_mov,
            max_action = delta_mov,
            max_episode_length = args.episode_length)
        
        design = DesignBestArmLinearBanditNoDenominatorContinuous(
            env = env,
            lambd = lambd,
            sigma = sigma,
            embedding = embedding,
            num_of_maximizers = args.num_maximizers,
        )

        # convex_solver = FrankWolfe(env, objective=design, num_components = 1, solver = AdditiveGradient, SummarizedPolicyType = MixturePolicy, verbosity = 4)
        convex_solver = InteriorPoint(env, objective=design)
        # define the feedback class
        feedback = ContinuousBanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, video = False, markovian = args.episodic_feedback, maximization_set_method = 'thompson_sampling')
        
    
    elif args.env_type == 'discrete':
        env = ConstrainedMaxMovement(
            action_space=embedded_action_space,
            action_space_pre_embedding=action_space,
            theta_star=theta_star,
            sigma=sigma,
            delta=delta_mov,
            max_episode_length = args.episode_length)
        
        design = DesignBestArmLinearBanditNoDenominator(
            env = env,
            lambd=lambd,
            sigma=sigma,
            init_ucb = 0.6
        )

        if args.policy == 'density':
            SummarizedPolicyType = DensityPolicy
        elif args.policy == 'marginal':
            SummarizedPolicyType = MarginalDensityPolicy

        # define the convex solver
        convex_solver = FrankWolfe(env, objective=design, num_components = args.num_components, solver = DP, SummarizedPolicyType = SummarizedPolicyType)
        # define the feedback class
        feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, video = False, markovian = args.episodic_feedback)


    else:
        # warn of invalid environment type
        raise ValueError('Invalid environment type')
    
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

    # if discrete
    if args.env_type == 'discrete':
        # save the results
        x_evaluations = action_space[visitations[0][0]]
        f_evaluations = []
        for x in x_evaluations:
            f_evaluations.append(theta_star(x.reshape(1, -1)))
        
        best_guesses = action_space[feedback.best_arm]

        file_name = 'experiments/flow_ode/results/' + args.env_type  + f'/delta_mov_{args.delta_mov}/noise_{args.noise}/num_features_{args.num_features}/'

    print('done')