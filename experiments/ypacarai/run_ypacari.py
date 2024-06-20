from enum import Enum
import random

import numpy as np
import torch
from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import argparse

from mdpexplore.solvers.dp import DP
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator, DesignBestArmLinearBanditEIDummy, GreedyEIDummy
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.feedback.bandit_feedback import BanditFeedback

from experiments.ypacarai.fit_ypacarai import Schekel2D, YpacaraiEnv
import os

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='Constrained Ypacari Problem.')
    # arguments I know I will need
    parser.add_argument('--seed', default=121, type=int, help='Use this to set the seed for the random number generator')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--episodic_feedback', default=False, type=bool, help='Wether we use episodic feedback or not')
    parser.add_argument('--episode_length', default=50, type=int, help='Length of the episode')
    parser.add_argument('--noise', default=0.001, type=float, help='Noise variance')
    parser.add_argument('--num_features', default=100, type=int, help='Number of features')
    parser.add_argument('--mix_objective', default=False, type=bool, help='Wether we mix our DoE design with UCB')
    parser.add_argument('--num_components', default=1, type=int, help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=3, type=int, help='Number of episodes')
    parser.add_argument('--EI', default=False, type=bool, help='Wether we use EI or not')
    parser.add_argument('--greedyEI', default=False, type=bool, help='Wether we use greedy EI or not')
    parser.add_argument('--video', default=False, type=bool, help='Wether we want to save a video')
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
    scheckel_function = Schekel2D()
    theta_star = lambda x: scheckel_function.query_function(x.reshape(-1, 2)).reshape(-1).item()
    # set noise level
    sigma = np.sqrt(args.noise)

    # set regularization parameter
    lambd = 1.0

    # define the kernel

    # define the action space
    action_space = np.load('ypacarai_centroids.npy')

    if args.video:
        ucbs = np.empty(shape = (0, action_space.shape[0]))
        lcbs = np.empty(shape = (0, action_space.shape[0]))
        means = np.empty(shape = (0, action_space.shape[0]))
        actions = np.empty(shape = (0, 1))

        np.save('ucb.npy', ucbs)
        np.save('lcb.npy', lcbs)
        np.save('mean.npy', means)
        np.save('actions.npy', actions)

    # define the initial state
    init_state = 59
    # define the terminal state
    # terminal_state = 43
    terminal_state = 59

    kernel_rbf = KernelFunction(kernel_name='squared_exponential', gamma = 0.2, d = 2)

    # now define the nystrom embedding
    embedding = NystromFeatures(m = torch.tensor(args.num_features), kernel_object=kernel_rbf)
    embedding.fit_gp(torch.tensor(action_space), None)
    # define the embedded action space for the discrete case
    embedded_action_space = embedding.embed(torch.tensor(action_space)).detach().numpy()

    # finally define the estimator
    estimator = KernelizedFeatures(embedding, m = args.num_features, s = sigma, lam = lambd, d = 2, diameter = 0.5)

    # define the environment and reward function
    adjacency_matrix = np.load('ypacarai_adjacency_matrix.npy')

    env = YpacaraiEnv(
        adjacency_matrix=adjacency_matrix,
        action_space=embedded_action_space,
        action_space_pre_embedding=action_space,
        theta_star=theta_star,
        sigma=sigma,
        max_episode_length = args.episode_length,
        init_state = init_state,
        terminal_state = terminal_state)
    
    if args.EI:
        design = DesignBestArmLinearBanditEIDummy(
            env = env,
            init_ucb = 0.0
        )
    elif args.greedyEI:
        design = GreedyEIDummy(
            env = env,
            init_ucb = 0.0
        )
    else:
        design = DesignBestArmLinearBanditNoDenominator(
            env = env,
            lambd=lambd,
            sigma=sigma,
            init_ucb = 1.0
        )

    # choose the policy
    if args.policy == 'density':
        SummarizedPolicyType = DensityPolicy
    elif args.policy == 'marginal':
        SummarizedPolicyType = MarginalDensityPolicy

    # define the convex solver
    convex_solver = FrankWolfe(env, objective=design, num_components = args.num_components, solver = DP, SummarizedPolicyType = SummarizedPolicyType)
    # define the feedback class
    if args.EI or args.greedyEI:
        update_mean = True
    else:
        update_mean = False
    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, video = args.video, markovian = args.episodic_feedback, update_mean = update_mean)

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
    x_evaluations = action_space[visitations[0][0]]
    f_evaluations = []
    for x in x_evaluations:
        f_evaluations.append(theta_star(x.reshape(1, -1)))
    
    best_guesses = action_space[feedback.best_arm]

    if args.episodic_feedback:
        file_name = f'experiments/ypacarai/results/episodic_feedback/noise_var_{args.noise}/num_features_{args.num_features}/num_episodes_{args.episodes}/episode_length_{args.episode_length}/'
    else:
        file_name = f'experiments/ypacarai/results/immediate_feedback/noise_var_{args.noise}/num_features_{args.num_features}/num_episodes_{args.episodes}/episode_length_{args.episode_length}/'
    
    if args.EI:
        algo_name = '/EI'
    elif args.greedyEI:
        algo_name = '/greedyEI'
    else:
        algo_name = f'/MDPExplore/policy_type_{args.policy}/num_components_{args.num_components}'

    file_name = file_name + algo_name + f'/seed_{args.seed}/'

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # save the results
    np.save(file_name + 'x_evaluations.npy', x_evaluations)
    np.save(file_name + 'f_evaluations.npy', f_evaluations)
    np.save(file_name + 'best_guesses.npy', best_guesses)