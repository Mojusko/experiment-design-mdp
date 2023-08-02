
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP 

from mdpexplore.env.time_chain import TimeChain
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.env.grid_worlds import DummyGridWorld
from mdpexplore.mdpexplore import MdpExplore
import argparse


from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.policy_generator import PolicyGenerator
from mdpexplore.functionals.doe_adaptive_functionals import *


parser = argparse.ArgumentParser(description='Gridworlds Problem.')
parser.add_argument('--seed', default=12, type=int,
                    help='Use this to set the seed for the random number generator')
parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
parser.add_argument('--cores', default=None, type=int, help='number of cores')
parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
parser.add_argument('--policy', default='density', type=str,
                    help='Summarized policy type (mixed/average/density)')
parser.add_argument('--num_components', default=10000, type=int,
                    help='Number of MaxEnt components (basic policies)')
parser.add_argument('--episodes', default=1, type=int, help='Number of evaluation policy unrolls')
parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
parser.add_argument('--adaptive', default="Bayes", type=str, help='Number of repeats')
parser.add_argument('--opt', default="false", type=str, help='Number of repeats')
parser.add_argument('--linesearch', default='line-search', type=str, help="type")
parser.add_argument('--savetrajectory', default=None, type=str, help="type")

args = parser.parse_args()

if args.policy == 'mixed':
    args.policy = MixturePolicy
elif args.policy == 'average':
    args.policy = AveragePolicy
elif args.policy == 'density':
    args.policy = DensityPolicy
else:
    raise ValueError('Invalid policy type')

env = DummyGridWorld(max_episode_length=20)
design = AdaptiveDesignD(env, lambd=1e-3)

me = MdpExplore(
    env,
    objective=design,
    solver=DP,
    step=None,
    method='cvxpy',
    verbosity=args.verbosity,
)

val, opt_val = me.run(
    num_components=args.num_components,
    episodes=args.episodes,
    SummarizedPolicyType=args.policy,
    accuracy=args.accuracy,
    save_trajectory=args.savetrajectory
)
print (me.policies, len(me.policies))
vals = np.array(val)
