import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, LatticeEmbedding
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from mdpexplore.solvers.lp import LP
from sklearn.cluster import KMeans
from scipy.linalg import null_space, orth
from mdpexplore.env.time_chain import TimeChain
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.tracking_policy import TrackingPolicy

from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.functionals.reward_functional import DesignBayesD, DesignBayesC, DesignC
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.mdpexplore import MdpExplore
from scipy.integrate import odeint
from stpy.helpers.helper import cartesian
import argparse

from fit_pharmaco import *

a = 5.
b = 10.
c = 10.

gamma = 0.05
mu_freq = 50

torch.manual_seed(5)
np.random.seed(5)

lam = 5e-1
sigma = 0.05
k = 8
n = 128
m = 150
dt = 0.01

a_values = [4, 6]
b_values = [9, 11]
c_values = [9, 11]

parameter_families = cartesian([a_values, b_values, c_values])


def eval(t, u0=1., v0=0., mu=1, k=0.3, sigma=0.0):
    def pend(y, t, k):
        theta, omega = y
        dydt = [-a * theta, b * theta - c * omega]
        return dydt

    init = [u0, v0]
    sol = odeint(pend, init, t.view(-1), args=(k,))
    return torch.from_numpy(sol[:]) + sigma * torch.randn(size=(t.size())).double()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Beilschmiedia Problem.')
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
    parser.add_argument('--uncertain', default="false", type=str, help="type")
    parser.add_argument('--savetrajectory', default=None, type=str, help="type")
    parser.add_argument('--measurements', default=5, type=int, help="type")
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

    N = int(args.measurements)

    emb_blood = HermiteEmbedding(m=m, gamma=gamma, kappa=1)
    emb_stomach = HermiteEmbedding(m=m, gamma=gamma, kappa=1)

    emb_noise = HermiteEmbedding(m=m, gamma=gamma, kappa=0.05)
    embed_noise_non_stationary = lambda x: torch.diag(torch.exp(-3 * x).view(-1)) @ emb_noise.embed(x)
    emb_noise_non_stationary = CustomEmbedding(d=1, embedding_function=embed_noise_non_stationary, m=m)

    embed_final = lambda x: torch.vstack([emb_stomach.embed(x), emb_blood.embed(x), emb_noise_non_stationary.embed(x)])
    emb_final = CustomEmbedding(d=1, embedding_function=embed_final, m=3 * m)

    t = torch.linspace(0, 1, n, requires_grad=True).view(-1, 1).double()

    patient_theta = torch.randn(size=(m, 1)).double() * 0.1

    g = lambda x: emb_noise_non_stationary.embed(x).detach() @ patient_theta

    f = eval(t.detach(), k=k, sigma=0)
    f_noisy = eval(t.detach(), k=k, sigma=sigma)

    GP = KernelizedFeatures(embedding=emb_blood, m=m, s=sigma, lam=lam)
    Phi = emb_blood.embed(t)
    Phi_noise = emb_noise_non_stationary.embed(t)

    Cs = []
    for a, b, c in parameter_families:
        _, constraints_full = constraint_operator(a, b, c, emb_blood, t)
        # # C acts on stomatch, blood
        C = torch.from_numpy(null_space(constraints_full.detach(), rcond=1e-7)).T.detach()
        # we need to append the specific variations
        C = torch.hstack([C, torch.zeros(size=(C.size()[0], m)).double()]).numpy()
        Cs.append(C)

    env = TimeChain(emb_final.embed, time_period=n, max_events=N, min_event_distance=3, dt=1. / N, max_episode_length=n)

    # average variant for simplicity
    C = sum(Cs) / len(Cs)

    if args.adaptive == "Bayes":
        design = DesignBayesC(env, lambd=1.0, scale_reg=False, sigma=sigma, C = Cs)
    else:
        design = DesignC(env, lambd=1.0, C=Cs, sigma = sigma)

    initial_policy = False

    if args.random == "true":
        initial_policy = True
        args.num_components = 1

    me = MdpExplore(
        env,
        objective=design,
        solver=LP,
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
        if args.uncertain == "false":
            np.savetxt("results/opt.txt", np.array([opt_val]))
        else:
            np.savetxt("results/un-opt.txt", np.array([opt_val]))
