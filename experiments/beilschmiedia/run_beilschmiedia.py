import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

torch.manual_seed(24)
torch.use_deterministic_algorithms(True)
np.random.seed(24)
random.seed(24)
import os
from sklearn.cluster import AgglomerativeClustering
from stpy.helpers.transformations import transform
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.kernels import KernelFunction
from sensepy.benchmarks.spatial_problem import SpatialProblem
from sensepy.benchmarks.bels.bels_problem import BeilschmiediaProblem
from stpy.continuous_processes.gauss_procc import GaussianProcess

from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP
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
#from mdpexplore.convex_solvers.cyipopt import InteriorPoint

# functionals
from mdpexplore.functionals.doe_adaptive_functionals import AdaptiveDesignD
from mdpexplore.functionals.doe_static_functionals import DesignD

# environments
from mdpexplore.env.grid_worlds import DummyGridWorld
from mdpexplore.env.stochastic_grid_world import StochasticGridWorld, StochasticDummyGridWorld
from mdpexplore.env.quad_tree_env import QuadTreeGrid

# feedbacks
from mdpexplore.feedback.feedback_base import EmptyFeedback, SimpleFeedback
from mdpexplore.feedback.poisson_feedback import PoissonFeedback

# general algorithm
from mdpexplore.mdpexplore import MdpExplore

import argparse
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
torch.use_deterministic_algorithms(True)
np.random.seed(24)
random.seed(24)
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
	parser.add_argument('--linesearch', default='line-search', type = str, help = "type")
	parser.add_argument('--uncertain', default="false", type = str, help = "type")
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

	torch.use_deterministic_algorithms(True)
	np.random.seed(24)
	random.seed(24)

	# Problem = BeilschmiediaProblem(m = 20, basis = "triangle",levels=5, gamma = 0.1, b = 0.5)
	# Problem.load_data(clusters = 100, prefix = "data/")
	# Problem.fit_model()

	if ~os.path.isfile("model.pt"):
		Problem = BeilschmiediaProblem(m = 20, basis = "triangle",levels=5, gamma = 0.1, b = 0.5)
		Problem.load_data(clusters = 100, prefix = "data/")
		Problem.fit_model()
		Problem.save_model("model.pt")

	Problem.load_saved_model("model.pt")
	p = Problem.return_process(n=20)
	# Problem.plot()

	torch.manual_seed(args.seed)
	torch.use_deterministic_algorithms(True)
	np.random.seed(args.seed)
	random.seed(args.seed)

	vol = np.min([action.volume() for action in Problem.hs2d.get_sets_level(5)])
	dt = 100 / (vol * Problem.b)
	print ("dt:", dt)
	integral = lambda S: Problem.estimator.packing.integral(S)*dt
	env = QuadTreeGrid(integral,Problem.hs2d,depth=5, max_episode_length = 64)

	basic_sets = [env.sets[i] for i in np.arange(0,env.states_num,1)]

	estimator = PoissonRateEstimator(None, Problem.hs2d, d=2, basis=Problem.basis,
									 feedback = 'histogram', kernel_object=Problem.kernel,
									 B=p.B, b=p.b, m=Problem.m, jitter=1e-5, opt='cvxpy', dual = False,
									 uncertainty = 'least-sq', approx = "ellipsoid")


	if args.adaptive == "Bayes":
		design = AdaptiveDesignD(env, lambd=1.0, sigma = 1.)
	else:
		design = DesignD(env, lambd=1.0, sigma = 1.)

	design.Sigma = np.sqrt(np.array([float(estimator.ucb(s, dt = dt)) for s in basic_sets]).reshape(-1))

	print (design.Sigma)

	if args.uncertain == "true":
		if args.opt != "true":
			feedback = PoissonFeedback(env, design, p, estimator = estimator, problem = Problem, opt = True)
		else:
			feedback = PoissonFeedback(env, design, p, estimator = estimator, problem = Problem, opt = False)
		design.Sigma_true = np.sqrt(np.array([float(Problem.estimator.mean_set(s, dt = dt)) for s in basic_sets]).reshape(-1))
	else:
		design.Sigma_true = design.Sigma
		feedback = EmptyFeedback(env, design)


	print ("True Sigma:")
	print (design.Sigma_true)
	print ("========================")

	initial_policy = False

	if args.random == "true":
		initial_policy = True
		args.num_components = 1


	def run_single(_):
		convex_solver = FrankWolfe(env, objective=design,
			num_components = args.num_components,
			solver = DP,
			SummarizedPolicyType = DensityPolicy,
			accuracy = args.accuracy)


		me = MdpExplore(
			env=env,
			objective=design,
			convex_solver=convex_solver,
			verbosity=args.verbosity,
			feedback=feedback,
			general_policy = 'markovian'
		)

		val, opt_val = me.run(
			episodes=args.episodes,
			#save_trajectory=args.savetrajectory,
			#return_visitations=True
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
	np.savetxt(args.save,vals)
	if args.opt == "true":
		if args.uncertain == "false":
			np.savetxt("results/opt.txt",np.array([opt]))
		else:
			np.savetxt("results/un-opt.txt", np.array([opt]))