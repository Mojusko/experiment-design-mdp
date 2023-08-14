import numpy as np
import torch
from mdpexplore.mdpexplore_con import ContinuousMdpExplore
from mdpexplore.env.linear_system import LinearSystem
from mdpexplore.env.linear_system import LinearPolicy
from mdpexplore.utils.continuous_reward_functionals import ContinuousDesignBayesD
from mdpexplore.solvers.gradient import GradientSolver
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
#from stpy.embeddings.embedding import HermiteEmbedding
from mdpexplore.utils.embedding import HermiteEmbedding, Embedding

if __name__ == "__main__":
    gamma = 0.1
    m = 16
    d = 1
    emb = HermiteEmbedding(m=m, gamma=gamma, kappa=1, d = d)




    env = LinearSystem(emb = emb, dim = d, sigma = 0.0001, max_episode_length = 3)
    objective = ContinuousDesignBayesD(env)
    solver = GradientSolver

    me = ContinuousMdpExplore(env, objective, solver, verbosity=4)
    num_components = 1
    me.run(num_components, episodes=5, accuracy= None, SummarizedPolicyType=MixturePolicy)
    #print (me.trajectory)
    print ("Policies")
    print ([policy.K for policy in me.policies])
