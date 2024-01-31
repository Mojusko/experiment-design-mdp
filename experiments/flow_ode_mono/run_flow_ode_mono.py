import random
import os

import numpy as np
import torch
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import argparse

from mdpexplore.solvers.dp import DP
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditEIDummy, DesignBestArmLinearBanditNoDenominator
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.feedback.bandit_feedback import BanditFeedback

import matplotlib.pyplot as plt

from experiments.flow_ode.fit_flow_ode import ode_kernel, SchreckerODE, ode_embedding
from experiments.flow_ode_mono.fit_flow_ode_mono import ODEMonoEnv

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='Schrecker Monotonic ODE Problem.')
    # arguments I know I will need
    parser.add_argument('--seed', default=1, type=int, help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--episodic_feedback', default=False, type=bool, help='Wether we use episodic feedback or not')
    parser.add_argument('--episode_length', default=10, type=int, help='Length of the episode')
    parser.add_argument('--noise', default=0.0001, type=float, help='Noise variance')
    parser.add_argument('--delta_mov', default=0.15, type=float, help='Maximum movement constraint')
    parser.add_argument('--num_features', default=121, type=int, help='Number of features')
    parser.add_argument('--EI', default=False, type=bool, help='Wether we use EI or not')
    parser.add_argument('--num_components', default=1, type=int, help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=10, type=int, help='Number of episodes')
    parser.add_argument('--plot', default=False, type = bool, help = "Wether we should save the paths and the potential maximizers for plotting")
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

    # define the kernel
    # generate a 2 by 2 uniform grid of points, then embed them in the 2d space
    grid_1d = np.linspace(-0.5, 0.5, 11)
    grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
    action_space = grid

    # create a finer grid for the nyström embedding
    grid_1d = np.linspace(-0.5, 0.5, 11)
    grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
    nystrom_space = grid

    prior_var = 0.0225
    embedding = ode_embedding(num_features = args.num_features, action_space = nystrom_space, alpha_ode = 1.0, alpha_rbf = 0.001, ard = False)
    # define the embedded action space for the discrete case
    embedded_action_space = embedding.embed(torch.tensor(action_space)).detach().numpy()
    # define the prior mean and variance of the ode model
    prior_mean = (embedded_action_space[:, 0] * 0.6)
    prior_var = 0.0225
    # redefine the embedding
    embedding = ode_embedding(num_features = args.num_features, action_space = nystrom_space, alpha_ode = prior_var, alpha_rbf = 0.001, ard = False)

    # finally define the estimator
    estimator = KernelizedFeatures(embedding, m = args.num_features, s = sigma, lam = lambd, d = 2, diameter = 0.5)
    
    env = ODEMonoEnv(
        action_space=embedded_action_space,
        action_space_pre_embedding=action_space,
        theta_star=theta_star,
        sigma=sigma,
        delta=delta_mov,
        max_episode_length = args.episode_length,
        init_state = 0
    )
    
    if args.EI:
        design = DesignBestArmLinearBanditEIDummy(
            env = env,
            init_ucb = 0.0,
            prior_mean = prior_mean
        )
    else:
        design = DesignBestArmLinearBanditNoDenominator(
            env = env,
            lambd=lambd,
            sigma=sigma,
            init_ucb = 0.6
        )
    
    # define the policy type
    if args.policy == 'density':
        policy_type = DensityPolicy
    elif args.policy == 'marginal':
        policy_type = MarginalDensityPolicy
    
    # define the convex solver
    convex_solver = FrankWolfe(env, objective=design, num_components = args.num_components, solver = DP, SummarizedPolicyType = policy_type)
    # define the feedback class
    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, video = False, markovian = args.episodic_feedback, update_mean = args.EI, prior_mean = prior_mean)
    
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

    if args.plot:
        # save the paths, and the potential maximizers
        paths = np.zeros((args.episodes, args.episode_length + 1, 2))
        for ep_idx in range(args.episodes):
            paths[ep_idx] = action_space[visitations[ep_idx][0]]
        np.save('flow_mono_paths.npy', paths)
        # save the ucbs and lcbs
        np.save('flow_mono_ucbs.npy', design.ucbs)
        np.save('flow_mono_lcbs.npy', design.lcbs)

    # save the results
    x_evaluations = np.zeros((args.episodes, args.episode_length, 2))
    for ep in range(args.episodes):
        x_evaluations[ep, :] = action_space[visitations[ep][1:]]
    
    x_evaluations = x_evaluations.reshape(args.episodes * args.episode_length, 2)

    f_evaluations = []
    for x in x_evaluations:
        f_evaluations.append(theta_star(x.reshape(1, -1)))
    
    best_guesses = action_space[feedback.best_arm]

    file_name = f'experiments/flow_ode_mono/results/delta_mov_{args.delta_mov}/noise_{args.noise}/num_features_{args.num_features}/'

    if args.episodic_feedback:
        file_name = f'experiments/flow_ode_mono/results/episodic_feedback/delta_mov_{args.delta_mov}/noise_var_{args.noise}/num_features_{args.num_features}/num_episodes_{args.episodes}/episode_length_{args.episode_length}/'
    else:
        file_name = f'experiments/flow_ode_mono/results/immediate_feedback/delta_mov_{args.delta_mov}/noise_var_{args.noise}/num_features_{args.num_features}/num_episodes_{args.episodes}/episode_length_{args.episode_length}/'
    
    if args.EI:
        algo_name = '/EI'
    else:
        algo_name = f'/MDPExplore/policy_type_{args.policy}/num_components_{args.num_components}'

    file_name = file_name + algo_name + f'/seed_{args.seed}/'

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # save the results
    np.save(file_name + 'x_evaluations.npy', x_evaluations)
    np.save(file_name + 'f_evaluations.npy', f_evaluations)
    np.save(file_name + 'best_guesses.npy', best_guesses)