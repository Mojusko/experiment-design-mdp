import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP
from sklearn.cluster import KMeans
from scipy.linalg import null_space, orth
from mdpexplore.env.time_chain import TimeChain
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.policies.tracking_policy import TrackingPolicy
from mdpexplore.utils.reward_functionals import DesignBayesD, DesignBayesC, DesignC, DesignD
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.env.grid_worlds import DummyGridWorld
from mdpexplore.env.stochastic_grid_world import StochasticGridWorld, StochasticDummyGridWorld
from mdpexplore.mdpexplore import MdpExplore
from scipy.integrate import odeint
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gridworlds Problem.')
    parser.add_argument('--seed', default=12, type=int,
                        help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--cores', default=None, type=int, help='number of cores')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
    parser.add_argument('--policy', default='density', type=str,
                        help='Summarized policy type (mixed/average/density)')
    parser.add_argument('--num_components', default=10, type=int,
                        help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=4, type=int, help='Number of evaluation policy unrolls')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
    parser.add_argument('--adaptive', default="Bayes", type=str, help='Number of repeats')
    parser.add_argument('--opt', default="false", type=str, help='Number of repeats')
    parser.add_argument('--linesearch', default='line-search', type=str, help="type")
    parser.add_argument('--savetrajectory', default=None, type=str, help="type")
    parser.add_argument('--probability', default=0.9, type=float, help="type")
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

    env = StochasticDummyGridWorld(prob = args.probability)
    #env = DummyGridWorld(prob = args.probability, max_episode_length=20)

    if args.adaptive == "Bayes":
        design = DesignBayesD(env, lambd=1e-3)
    else:
        design = DesignD(env, lambd=1e-3)

    initial_policy = False

    if args.random == "true":
        initial_policy = True
        args.num_components = 1

    me = MdpExplore(
        env,
        objective=design,
        solver=DP,
        step=args.linesearch,
        method='frank-wolfe',
        verbosity=args.verbosity,
        initial_policy=initial_policy

    )

    val, opt_val = me.run(
        num_components=args.num_components,
        episodes=args.episodes,
        SummarizedPolicyType=args.policy,
        accuracy=args.accuracy,
        save_trajectory=args.savetrajectory
    )
    vals = np.array(val)
    np.savetxt(args.save, vals)
    if args.opt == "true":
        np.savetxt("results/opt+"+str(np.round(args.probability,2))+".txt", np.array([opt_val]))
