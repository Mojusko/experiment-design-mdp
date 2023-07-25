import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import AgglomerativeClustering
from stpy.helpers.transformations import transform
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.kernels import KernelFunction
from sensepy.benchmarks.spatial_problem import SpatialProblem
from stpy.continuous_processes.gauss_procc import GaussianProcess
from mdpexplore.solvers.lp import LP
from sklearn.cluster import KMeans

from sensepy.benchmarks.bels.bels_problem import BeilschmiediaProblem
from mdpexplore.env.quad_tree_env import QuadTreeGrid
from mdpexplore.functionals.reward_functionals import DesignBayesD
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.mdpexplore import MdpExplore


if __name__ == "__main__":
	depth = 5
	torch.manual_seed(24)
	torch.use_deterministic_algorithms(True)
	np.random.seed(24)
	random.seed(0)

	Problem = BeilschmiediaProblem(m = 20, basis = "triangle", levels = depth, b = 0.5, gamma = 0.1)
	Problem.load_data(clusters = 50, prefix = "data/")
	Problem.fit_model()
	p = Problem.return_process(n=20)

	actions = Problem.hs2d.get_sets_level(depth)
	vol = np.min([action.volume() for action in actions])
	dt = 1. / (vol * Problem.b)
	print ("dt:", dt)
	integral = lambda S: Problem.estimator.packing.integral(S)*dt

	env = QuadTreeGrid(integral,Problem.hs2d, depth = depth)
	design = DesignBayesD(env, lambd=1.0)

	basic_sets = [env.sets[i] for i in np.arange(0,env.states_num,1)]

	estimator = PoissonRateEstimator(None, Problem.hs2d, d=2, basis=Problem.basis,
									 feedback = 'histogram', kernel_object=Problem.kernel,
									 B=p.B, b=p.b, m=Problem.m, jitter=1e-5, opt='cvxpy', dual = False,
									 uncertainty = 'least-sq', approx = "ellipsoid")
	design.Sigma = np.array([float(estimator.ucb(s)) for s in basic_sets]).reshape(-1)
	print (design.Sigma)

	def callback(state_history, objective):
		data = []
		state = state_history[-1]
		S = env.sets[state]
		y = p.sample(S)
		n = torch.randn(size = (y.size()[0],Problem.estimator.get_m())).double() if y is not None else None
		datapoint = (S, n ,dt)

		estimator.add_data_point(datapoint)
		estimator.fit_gp()
		objective.Sigma = np.array([float(estimator.ucb(s)) for s in basic_sets]).reshape(-1)


	# run clustering on states
	emb = np.concatenate(([integral(S).view(1, -1) for S in actions]))
	kmeans = KMeans(n_clusters=10, random_state=0).fit(emb)
	classes = kmeans.predict(emb)

	transformed_sector_map = classes.reshape(env.height, env.width)

	fig, ax = Problem.plot(show = False, points = True, alpha = 0.8, colorbar=True)
	plt.savefig("../figs/map.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)


	fig, ax = Problem.plot(show = False, points = False, alpha = 0.2, colorbar=False, levels = depth)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(transformed_sector_map, cmap='tab20', extent=[-1,1,-1,1])
	plt.savefig("../figs/clustering.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)


	plt.show()
	#
	# me = MdpExplore(
	# 	env,
	# 	objective=design,
	# 	solver=LP,
	# 	step=None,
	# 	method='frank-wolfe',
	# 	verbosity=3,
	# 	callback = callback
	# )
	#
	# val, opt_val = me.run(
	# 	num_components=1,
	# 	episodes=256,
	# 	SummarizedPolicyType=DensityPolicy,
	# 	accuracy=0.05,
	# 	plot=False,
	# )