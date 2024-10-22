import torch
import random
import torch
import argparse
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.candidate_set import CandidateDiscreteSet
from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures

from mutedpy.utils.sequences.sequence_utils import generate_all_combination, from_mutation_to_variant, from_variant_to_integer
from mutedpy.protein_learning.active_learning.generate_predictions import load_model
from mutedpy.experiments.streptavidin.streptavidin_loader import load_everything_we_have, load_first_round

from doexpy.env.bandits import Bandits
from doexpy.functionals.doe_static_functionals import DesignA
from doexpy.mdpexplore import MdpExplore
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

parser = argparse.ArgumentParser(description='Protein capacity experiment.')
parser.add_argument('--T', default=2, type=int, help='Name of the file')
args = parser.parse_args()

# load the model 
model_params = 'params/esm2linearfinal_model_params.p'
GP, embed =  load_model(model_params,"")
sigma = GP.s

# load the data
x, y, dts = load_first_round()
phi = embed(x)[:,0,:]
n = phi.size()[0]

if len(phi.size())>2:
    phi = phi[:,0,:]

# # Calculate Nystom Embeddings on Phi
Nystrom = NystromFeatures(GP.kernel_object, m = None, approx = 'svd-explained')
# 90% of explained variance 
Nystrom.fit_gp(phi, y, explained_variance = 1.)
embed2 = lambda x: Nystrom.embed(x)
phi2 = embed2(phi)

# phi2 = phi2/sigma

phi2 = phi/sigma

# the budget 
T = args.T

env = Bandits(
    action_space=phi2
    )

design = DesignA(
    env=env,
    lambd=1.0,
    dim = 1,
    V = phi2.T@phi2
)

# define the convex solver
convex_solver = FrankWolfe(env,
                           objective=design,
                           num_components=2*phi.size()[1],
                           solver=DP,
                           step="line-search",
                           SummarizedPolicyType=DensityPolicy,
                           accuracy=1e-3)

# define the feedback class
feedback = EmptyFeedback(env, design)

initial_policy = False

Ts = torch.logspace(1,10,20,base=2)
env = Bandits(
action_space=phi2
)
me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver,
    verbosity=3,
    feedback=feedback,
    general_policy='markovian'
)
val, opt_val = me.run(
    episodes=int(T)
)
with open("results/output-linear.txt", "a") as f:
    f.write(f"{T},{-opt_val/T}\n")
