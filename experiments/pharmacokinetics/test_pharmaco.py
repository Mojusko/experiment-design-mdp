import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, LatticeEmbedding
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from doexpy.solvers.lp import LP
from sklearn.cluster import KMeans
from scipy.linalg import null_space, orth
from doexpy.env.time_chain import TimeChain
from doexpy.functionals.reward_functional import DesignBayesD, DesignBayesC, DesignC
from doexpy.policies.summary_policies.density_policy import DensityPolicy
from doexpy.mdpexplore import MdpExplore
from scipy.integrate import odeint
from stpy.helpers.helper import cartesian

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
N = 5

a_values = [4,6]
b_values = [9,11]
c_values = [9,11]

parameter_families = cartesian([a_values,b_values,c_values])


def eval(t, u0=1., v0=0., mu=1, k=0.3, sigma=0.0):
    def pend(y, t, k):
        theta, omega = y
        dydt = [-a * theta, b * theta - c * omega]
        return dydt

    init = [u0, v0]
    sol = odeint(pend, init, t.view(-1), args=(k,))
    return torch.from_numpy(sol[:]) + sigma * torch.randn(size=(t.size())).double()

if __name__ == "__main__":



    emb_blood = HermiteEmbedding(m=m, gamma=gamma, kappa=1)
    emb_stomach = HermiteEmbedding(m=m, gamma=gamma, kappa=1)

    emb_noise = HermiteEmbedding(m=m, gamma=gamma, kappa=0.05)
    embed_noise_non_stationary = lambda x: torch.diag(torch.exp(-3*x).view(-1)) @ emb_noise.embed(x)
    emb_noise_non_stationary = CustomEmbedding(d=1, embedding_function=embed_noise_non_stationary, m= m)

    embed_final = lambda x: torch.vstack([emb_stomach.embed(x), emb_blood.embed(x), emb_noise_non_stationary.embed(x)])
    emb_final = CustomEmbedding(d=1, embedding_function=embed_final, m=3 * m)

    t = torch.linspace(0, 1, n, requires_grad=True).view(-1, 1).double()

    patient_theta = torch.randn(size = (m,1)).double()*0.1

    g = lambda x: emb_noise_non_stationary.embed(x).detach()@patient_theta

    f = eval(t.detach(), k=k, sigma=0)
    f_noisy = eval(t.detach(), k=k, sigma=sigma)

    GP = KernelizedFeatures(embedding=emb_blood, m=m, s=sigma, lam=lam)
    Phi = emb_blood.embed(t)
    Phi_noise = emb_noise_non_stationary.embed(t)

    min_unc = None
    max_unc = None

    # evaluate a point
    index_t = np.linspace(0, n, N + 2)[1:N + 1].astype(int)

    t_point = t[index_t].view(N, 1).detach()
    y_point = f_noisy[index_t, 1] + g(t_point).view(-1)
    y_point_not_noisy = f[index_t, 1] + + g(t_point).view(-1)

    for a, b, c in parameter_families:
        (theta_constrained, theta_constrained2, theta_constrained3, std_traj) = fit_pharmaco_perturbed(a, b, c,
                                                                                                       t_point, y_point, GP, t, emb_blood,
                                                                                                       emb_noise_non_stationary, lam=lam, sigma=sigma)

        mu_constrained = Phi @ theta_constrained
        mu_constrained = mu_constrained.detach()

        mu_constrained2 = Phi @ theta_constrained2
        mu_constrained2 = mu_constrained2.detach()

        mu_constrained3 = Phi_noise @ theta_constrained3
        mu_constrained3 = mu_constrained3.detach()


        #plt.plot(t.detach(), mu_constrained, lw=2, alpha=0.5, color='tab:blue')
        plt.plot(t.detach(), mu_constrained2, lw=2, alpha=0.5, color='tab:purple')
        plt.plot(t.detach(), mu_constrained2+mu_constrained3, lw=2, alpha=0.5, color='tab:brown')

        if min_unc is None:
            min_unc = - std_traj * 2 + mu_constrained2.view(-1)
        else:
            min_unc = torch.minimum(- std_traj * 2 + mu_constrained2.view(-1), min_unc)
        if max_unc is None:
            max_unc = std_traj * 2 + mu_constrained2.view(-1)
        else:
            max_unc = torch.maximum(std_traj * 2 + mu_constrained2.view(-1), max_unc)

    # plt.plot(t.detach(),mu_constrained,lw= 2, alpha =0.5, color = 'tab:blue', label = 'est. $c_s$ (stomach)')
    plt.plot(t.detach(), mu_constrained2, lw=2, alpha=0.5, color='tab:purple', label='est. $c_b$ (blood)')
    plt.plot(t.detach(), mu_constrained2 + mu_constrained3, lw=2, alpha=0.5, color='tab:brown', label='est. $c_b$ (blood)')

    plt.plot(t.detach(), f[:, 0], label='true $c_s$ (stomach)', color='tab:green', lw=4)
    plt.plot(t.detach(), f[:, 1], label='true $c_b$ (blood)', color='tab:red', lw=4, zorder = 10)
    plt.plot(t.detach(), f[:, 1]+g(t).view(-1), label='true $c_b$ (blood patient)', color='tab:brown', lw=4)

    plt.plot(t_point,y_point,'ro', label = 'noisy measurements', ms = 5)
    plt.plot(t_point, y_point_not_noisy, 'ko', label='chosen measurements', ms = 5, zorder = 20)
    for index, q in enumerate(t_point):
        plt.plot([q, q], [y_point_not_noisy[index], y_point[index]], 'k--')

    plt.grid(linestyle="--", color='gray')
    plt.fill_between(t.detach().view(-1), min_unc.view(-1), max_unc.view(-1), color='tab:purple', alpha=0.5,
                     label='uncer. $c_b$ (due to model $\gamma$)')
    plt.ylabel("Concentrations", fontsize = 15)
    plt.xlabel("Time [days]", fontsize = 15)
    plt.legend(fontsize = 13, ncol=2, loc = "lower left", mode = 'expand',bbox_to_anchor=(-0.05,0.6,1.05,0.2))
    plt.savefig("../figs/pharmaco-demo.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)
    plt.show()


    Cs = []
    for a, b, c in parameter_families:
        _, constraints_full = constraint_operator(a, b, c, emb_blood, t)
        # # C acts on stomatch, blood
        C = torch.from_numpy(null_space(constraints_full.detach(), rcond=1e-10)).T.detach()
        # we need to append the specific variations
        C = torch.hstack([C,torch.zeros(size = (C.size()[0],m)).double()]).numpy()
        Cs.append(C)

    env = TimeChain(emb_final.embed, time_period = n,max_events = N, min_event_distance=3, dt = 1./N, max_episode_length=n)

    # average variant for simplicity
    C = sum(Cs)/len(Cs)
    #design = DesignC(env, lambd=1.0, C = C)
    design = DesignC(env, lambd=1.0, C = Cs)

    # me = MdpExplore(
    #     env,
    #     objective=design,
    #     solver=LP,
    #     step=None,
    #     method='frank-wolfe',
    #     verbosity=3,
    #     callback=None,
    # )
    #
    # val, opt_val = me.run(
    #     num_components=10,
    #     episodes=100,
    #     SummarizedPolicyType=DensityPolicy,
    #     accuracy=None,
    #     plot=False,
    # )
