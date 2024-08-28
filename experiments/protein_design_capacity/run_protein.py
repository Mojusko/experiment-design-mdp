import torch
#import numpy as np
import random
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import jax.numpy as np

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.candidate_set import CandidateDiscreteSet
from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures

from mutedpy.utils.sequences.sequence_utils import generate_all_combination, from_mutation_to_variant, from_variant_to_integer
from mutedpy.protein_learning.active_learning.generate_predictions import load_model
from mutedpy.experiments.streptavidin.streptavidin_loader import load_everything_we_have, load_first_round

from mdpexplore.env.bandits import Bandits
from mdpexplore.functionals.doe_static_functionals import DesignA
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.feedback.feedback_base import EmptyFeedback
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy


model_params = 'params/esm2final_model_params.p'
GP, embed =  load_model(model_params,"")
x, y, dts = load_first_round()
phi = embed(x)
n = phi.size()[0]

# Calculate Nystom Embeddings on Phi
Nystrom = NystromFeatures(GP.kernel_object, m = 100, approx = 'svd-explained')
Nystrom.fit_gp(phi, y, delta = 0.8)
embed2 = lambda x: Nystrom.embed(x)
phi2 = embed2(phi)
d = phi2.size()[1]
sigma = GP.s
phi2 = phi2/sigma
T = 1000

print ("Dim:", phi2.size())
# Calculate where we want to predict
# all 3-site mutants at 118, 119, 121, 3.2 milions of them
# parent = "T111T+S112S+"
# mutants = generate_all_combination([118,119,121],'NAK')
# mutants = [parent + s for s in mutants]
# print (mutants)
# variants = from_variant_to_integer(from_mutation_to_variant(mutants))
# phi_xtest = embed(variants)
# phi2_xtest = embed2(phi_xtest)

env = Bandits(
    action_space=phi2.numpy()
    )

design = DesignA(
    env=env,
    lambd=1.0,
    dim = 1
)

# define the convex solver
convex_solver = FrankWolfe(env,
                           objective=design,
                           num_components=T,
                           solver=DP,
                           step="line-search",
                           SummarizedPolicyType=DensityPolicy,
                           accuracy=1e-3)

# define the feedback class
feedback = EmptyFeedback(env, design)

initial_policy = False

me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver,
    verbosity=3,
    feedback=feedback,
    general_policy='markovian'
)

val, opt_val = me.run(
    episodes=1
)

print (-opt_val*T)



