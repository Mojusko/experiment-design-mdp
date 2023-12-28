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

from mdpexplore.solvers.ddpg import DDPG
from mdpexplore.solvers.additive_gradient import AdditiveGradient
from mdpexplore.solvers.dp import DP
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.convex_solvers.cyipopt import InteriorPoint
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.env.bandits import Bandits, Bandits_Left_Right, ConstrainedMaxMovement
from mdpexplore.env.continuous_bandits import ContinuousMovementConstrainedBayesianOptimization
from mdpexplore.functionals.bandit_functionals import DesignRewardBandit, DesignBestArmLinearBandit, DesignBestArmLinearBanditNoDenominator, DesignBestArmLinearBanditNoDenominatorContinuous
from mdpexplore.functionals.doe_adaptive_functionals import AdaptiveDesignD
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from mdpexplore.feedback.bandit_feedback import BanditFeedback, ContinuousBanditFeedback

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

from scipy.stats.qmc import Sobol
from scipy.integrate import odeint
import time

from experiments.flow_ode.fit_flow_ode import ode_kernel, SchreckerODE, ode_embedding
from experiments.flow_ode_mono.fit_flow_ode_mono import ODEMonoEnv

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='Schrecker Monotonic ODE Problem.')
    # arguments I know I will need
    parser.add_argument('--seed', default=121, type=int, help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--episode_length', default=10, type=int, help='Length of the episode')
    parser.add_argument('--noise', default=0.0001, type=float, help='Noise variance')
    parser.add_argument('--delta_mov', default=0.15, type=float, help='Maximum movement constraint')
    parser.add_argument('--num_features', default=2048, type=int, help='Number of features')
    parser.add_argument('--mix_objective', default=False, type=bool, help='Wether we mix our DoE design with UCB')
    parser.add_argument('--num_components', default=1, type=int, help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=10, type=int, help='Number of episodes')
    # extra arguments
    parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
    parser.add_argument('--policy', default='density', type=str, help='Summarized policy type (mixed/average/density)')
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
    grid_1d = np.linspace(-0.5, 0.5, 201)
    grid = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)
    nystrom_space = grid

    # kernel_rbf = KernelFunction(kernel_name='squared_exponential', gamma = 0.05, d = 2, kappa = 0.002)
    # kernel_rbf = KernelFunction(kernel_name='ard', gamma = [0.1, 0.05], d = 2, kappa = 0.01)
    # kernel_ode_inner = ode_kernel()
    # kernel_ode = KernelFunction(kernel_function=kernel_ode_inner, kappa = 1.0, d = 2)

    # kernel = kernel_ode + kernel_rbf

    # # now define the nystrom embedding
    # embedding = NystromFeatures(m = torch.tensor(len(action_space)), kernel_object=kernel)
    # embedding.fit_gp(torch.tensor(action_space), None)
    embedding = ode_embedding(num_features = args.num_features, action_space = nystrom_space, alpha_ode = 1.0, alpha_rbf = 0.001, ard = False)
    # define the embedded action space for the discrete case
    embedded_action_space = embedding.embed(torch.tensor(action_space)).detach().numpy()

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
    
    design = DesignBestArmLinearBanditNoDenominator(
        env = env,
        lambd=lambd,
        sigma=sigma,
        mix_objectives=(args.mix_objective, 0.5),
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
    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, video = False, markovian = False)
    
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

    file_name = f'experiments/flow_ode_mono/results/delta_mov_{args.delta_mov}/noise_{args.noise}/num_features_{args.num_features}/'

# save the paths, and the potential maximizers
paths = np.zeros((args.episodes, args.episode_length+ 1, 2))
for ep_idx in range(args.episodes):
    paths[ep_idx] = action_space[visitations[ep_idx][0]]
np.save('flow_mono_paths.npy', paths)
# save the ucbs and lcbs
np.save('flow_mono_ucbs.npy', design.ucbs)
np.save('flow_mono_lcbs.npy', design.lcbs)

# plot the four paths
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(6, 6))

# contour plot of theta_star
x = np.linspace(-0.5, 0.5, 101)
y = np.linspace(-0.5, 0.5, 101)
X, Y = np.meshgrid(x, y)
Z = np.zeros((101, 101))

for i in range(100):
    for j in range(100):
        f = theta_star(np.array([X[i, j], Y[i, j]]).reshape(-1, 2))
        Z[i, j] = f

# plot the contour
contour_plot = ax[0].contourf(X, Y, Z, 20, cmap='Blues')
contour_plot = ax[0].contourf(X, Y, Z, 20, cmap='Blues')

# highlight the true best arm
real_func_values = [theta_star(act.reshape(-1, 2)) for act in action_space]
best_arm = action_space[np.argmax(real_func_values)]
ax[0].scatter(best_arm[0], best_arm[1], marker='*', color='k', s=200, label='True best arm')

for ep_idx in range(args.episodes):
    states = action_space[visitations[ep_idx][0]]
    ax[0].plot(states[:, 0], states[:, 1], 'o-', label=f'Episode {ep_idx + 1}')

# in the second axis plot the potential maximizers
mask = design.ucbs > np.max(design.lcbs)
potential_maximizers = action_space[mask]
ax[1].scatter(potential_maximizers[:, 0], potential_maximizers[:, 1], marker='o', color='green', label='Potential maximizers')

discarded_maximizers = action_space[np.logical_not(mask)]
ax[1].scatter(discarded_maximizers[:, 0], discarded_maximizers[:, 1], marker='o', color='red', label='Discarded maximizers')

# plot the legend
ax[0].legend()
ax[1].legend()

plt.show()


print('done')