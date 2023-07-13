from enum import Enum
import random

import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
import argparse
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from mdpexplore.solvers.min import MinSolver
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.env.bandits import Bandits
from mdpexplore.utils.reward_functionals import DesignRewardBandit, DesignBestArmLinearBandit
from mdpexplore.policies.density_policy import DensityPolicy

if __name__ == "__main__":

    class Type(str, Enum):
        reward = 'REWARD'
        best_arm_linear = 'BEST_ARM_LINEAR'


    parser = argparse.ArgumentParser(description='Bandits.')
    parser.add_argument('--seed', default=12, type=int,
                        help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='Name of the file')
    parser.add_argument('--cores', default=None, type=int, help='Number of cores')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
    parser.add_argument('--type', default='REWARD', type=Type, choices=list(Type),
                        help='One of 2 possible values: `REWARD` or `BEST_ARM_LINEAR`')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    theta_star = np.array([1, 1])
    action_space = np.array([
        [1, 0],
        [0, 1],
        [np.sqrt(0.5), np.sqrt(0.5)],
    ])
    sigma = 1
    delta = 1
    lambd = 1 / 50

    env = Bandits(
        action_space=action_space,
        theta_star=theta_star,
        sigma=sigma)

    kernel = KernelFunction(kernel_name='linear')
    estimator = GaussianProcess(kernel=kernel)

    if args.type == 'REWARD':

        design = DesignRewardBandit(
            action_space_size=action_space.shape[0]
        )


        def callback(trajectory, objective):
            action = trajectory[-1]
            z = action_space[action]
            eps = np.random.normal(0, sigma)
            fun_value = z @ theta_star + eps
            estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                     torch.tensor([[fun_value]]))
            estimator.fit()
            objective.mu = estimator.ucb(torch.tensor(action_space)).numpy().squeeze()

    elif args.type == 'BEST_ARM_LINEAR':

        design = DesignBestArmLinearBandit(
            action_space_size=action_space.shape[0],
            lambd=lambd
        )


        def callback(trajectory, objective):
            action = trajectory[-1]
            z = action_space[action]
            eps = np.random.normal(0, sigma)
            fun_value = z @ theta_star + eps
            estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                     torch.tensor([[fun_value]]))
            estimator.fit()
            objective.ucbs = estimator.ucb(torch.tensor(action_space)).numpy().squeeze()
            objective.lcbs = estimator.lcb(torch.tensor(action_space)).numpy().squeeze()

            # Compute the differences to get the UCBs and LCBs for the objective denominator
            n, m = action_space.shape
            arr1_reshaped = action_space.reshape(n, 1, m)
            arr2_reshaped = action_space.reshape(1, n, m)
            diffs = arr1_reshaped - arr2_reshaped
            objective.diff_ucbs = estimator.ucb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))
            objective.diff_lcbs = estimator.lcb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))
    else:
        raise NotImplementedError()


    def run_single(_):
        me = MdpExplore(
            env,
            objective=design,
            solver=MinSolver,
            method='frank-wolfe',
            verbosity=True,
            callback=callback)

        val, opt_val = me.run(
            num_components=1,
            episodes=20,
            SummarizedPolicyType=DensityPolicy,
        )
        return val, opt_val


    if args.cores is None:
        cores = mp.cpu_count() - 2
    else:
        cores = args.cores

    outputs = process_map(run_single, [repeat for repeat in range(args.repeats)], max_workers=cores)
    vals = []
    for repeat in range(args.repeats):
        val, opt_val = outputs[repeat]
        vals.append(val)
        if opt_val is not None:
            opt = opt_val
    vals = np.array(vals)
    np.savetxt(args.save, vals)
