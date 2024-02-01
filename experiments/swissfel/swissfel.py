import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stpy.helpers.helper import interval
# solvers
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP

# policy summarizations
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from mdpexplore.policies.summary_policies.average_policy import AveragePolicy
from mdpexplore.policies.summary_policies.tracking_policy import TrackingPolicy

# convex solvers
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
# from mdpexplore.convex_solvers.cyipopt import InteriorPoint

# functionals
from mdpexplore.functionals.doe_adaptive_functionals import AdaptiveDesignD, AdaptiveDesignHeteroD
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator
from mdpexplore.functionals.doe_static_functionals import DesignD

# environments
from mdpexplore.env.grid_worlds import DummyGridWorld
from mdpexplore.env.stochastic_grid_world import StochasticGridWorld, StochasticDummyGridWorld
from mdpexplore.env.global_grid_worlds import GlobalTransGridWorld
# feedbacks
from mdpexplore.feedback.feedback_base import EmptyFeedback, SimpleFeedback
from mdpexplore.feedback.bandit_feedback import BanditFeedback

# general algorithm
from mdpexplore.mdpexplore import MdpExplore
from scipy.integrate import odeint
import argparse

from stpy.test_functions.benchmarks import SwissFEL
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.continuous_processes.gauss_procc import GaussianProcess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gridworlds Problem.')
    parser.add_argument('--seed', default=12, type=int,
                        help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--cores', default=None, type=int, help='number of cores')
    parser.add_argument('--verbosity', default=4, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
    parser.add_argument('--policy', default='density', type=str,
                        help='Summarized policy type (mixed/average/density)')
    parser.add_argument('--num_components', default=1, type=int,
                        help='Number of MaxEnt components (basic policies)')

    parser.add_argument('--horizon', default=100, type=int, help='Number of evaluation policy unrolls')
    parser.add_argument('--episodes', default=1, type=int, help='Number of evaluation policy unrolls')
    parser.add_argument('--num_features', default=100, type=int, help='Number of evaluation policy unrolls')

    parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
    parser.add_argument('--worst', default="No", type=str, help='Number of repeats')
    parser.add_argument('--opt', default="false", type=str, help='Number of repeats')

    parser.add_argument('--linesearch', default='line-search', type=str, help="type")
    parser.add_argument('--savetrajectory', default=None, type=str, help="type")
    parser.add_argument('--random', default="false", type=str, help="type")

    args = parser.parse_args()

    if args.policy == 'mixed':
        args.policy = MixturePolicy
    elif args.policy == 'average':
        args.policy = AveragePolicy
    elif args.policy == 'density':
        args.policy = DensityPolicy
    elif args.policy == "tracking":
        args.policy = TrackingPolicy
    else:
        raise ValueError('Invalid policy type')
    sigma = 0.01
    size = 10
    height = size
    switch_weight = 200./size
    base_sigma = 1.
    env = GlobalTransGridWorld(size=height, max_episode_length=args.horizon)

    sigmas = np.zeros(shape=(size**2, size**2)) * sigma

    for i in range(size **2):
        for j in range(size **2):
            sigmas[i, j] = (np.sqrt((env.convert_to_grid(i)[0]-env.convert_to_grid(j)[0])**2 + (env.convert_to_grid(i)[1]-env.convert_to_grid(j)[1])**2)*switch_weight+base_sigma)*sigma

    print (sigmas)
    def sigma_fun(visitations):
        scaled = visitations/sigmas**2
        return scaled

    if args.worst == "Yes":
        design = DesignBestArmLinearBanditNoDenominator(
            env=env,
            lambd=1.,
            sigma= sigma * (base_sigma + switch_weight*np.sqrt((size ** 2 + size ** 2))),
            init_ucb=2.
        )
    else:

        design = DesignBestArmLinearBanditNoDenominator(
            env=env,
            lambd=1.,
            sigma= sigma * (base_sigma + switch_weight*np.sqrt((size ** 2 + size ** 2))),
            sigma_fun = sigma_fun,
            init_ucb=2.
        )

    # define the convex solver
    convex_solver = FrankWolfe(env, objective=design,
                               num_components=args.num_components,
                               solver=DP,
                               SummarizedPolicyType=DensityPolicy,
                               accuracy=args.accuracy)

    # define a swifel function
    Fel = SwissFEL(d=2, dts = 'evaluations.hdf5')
    F = lambda x: Fel.eval(torch.from_numpy(x)).numpy()

    # xtest
    action_space = interval(size, d = 2, L_infinity_ball=0.5)

    print ("grid size:")
    print (action_space.shape)

    # embeddings
    kernel = KernelFunction(kernel_name='squared_exponential', gamma = 0.3, d = 2, kappa = 1.)
    embedding = NystromFeatures(m=torch.tensor(size**2), kernel_object=kernel)
    embedding.fit_gp(torch.tensor(action_space), None)
    #estimator = KernelizedFeatures(embedding, m = torch.tensor(size**2), s = 1., lam = 1., d = 2)
    estimator = GaussianProcess(kernel=kernel, d=2)

    # fit the surogate model to get $\theta_star$
    y = F(action_space)
    estimator_true = KernelizedFeatures(embedding, m=torch.tensor(size**2), s = 1e-3, lam=1., d=2)
    estimator_true.add_points((torch.from_numpy(action_space),y))
    estimator_true.fit()
    F = lambda x: estimator_true.mean(torch.from_numpy(x))
    # define the feedback class

    sigma_fn_states = lambda s_coord,a_coord: (base_sigma+switch_weight*np.sqrt((s_coord[0]-a_coord[0])**2 +  (s_coord[1]-a_coord[1])**2))*sigma

    # define the embedded space
    env.action_space = embedding.embed(torch.tensor(action_space)).detach().numpy()
    env.emissions = embedding.embed(torch.tensor(action_space)).detach().numpy()

    if args.worst == "No":
        feedback = BanditFeedback(env, design, estimator, F, sigma = sigma * (base_sigma + switch_weight*np.sqrt((size ** 2 + size ** 2))), sigma_fn=sigma_fn_states, wort_case=False, prior_mean = 0)
    else:
        feedback = BanditFeedback(env, design, estimator, F, sigma = sigma * (base_sigma + switch_weight*np.sqrt((size ** 2 + size ** 2))), sigma_fn=sigma_fn_states, wort_case=True, prior_mean = 0)

    initial_policy = False

    if args.random == "true":
        initial_policy = True
        args.num_components = 1


    me = MdpExplore(
        env=env,
        objective=design,
        convex_solver=convex_solver,
        verbosity=args.verbosity,
        feedback=feedback,
        general_policy='non-markovian'
    )

    val, opt_val = me.run(
        episodes=args.episodes,
        save_trajectory=args.savetrajectory,
        # return_visitations=True
    )
    true_val = estimator_true.mean(torch.from_numpy(action_space))
    best_arm =torch.argmax(true_val)
    best_arms = [v for v in feedback.best_arm]
    regrets = [float(true_val[best_arm]-true_val[v]) for v in feedback.best_arm]

    print (regrets)

    vals = np.array(val)
    np.savetxt(args.save, np.array(regrets))