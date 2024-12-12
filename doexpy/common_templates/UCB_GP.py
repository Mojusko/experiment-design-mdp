import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np

from stpy.regression.kernelized_features import KernelizedFeatures
from stpy.embeddings.embedding import HermiteEmbedding

from doexpy.common_templates.Bandit import Bandit

warnings.filterwarnings("ignore",".*GUI is implemented.*")




class UCB_GP(Bandit):

	def __init__(self,x,
				 F,
				 estimator,
				 epsilon = 0.001,
				 opt = "discrete",
				 multistart = 25,
				 verbose = False,
				 beta = None,
				 delta = 0.01):
		"""
		Constructor of UCB-GP Algorithm
		:param x: intiail data, torch array 2 dim
		:param F: function that evaluates x -> y
		:param GP: covariance model
		:param epsilon: tolerance between |x_1 - x_2|<epsilon are considered the same
		:param opt:
				"general" - grid based optimization
				"first_order" - optimizer

		:param multistart: number of multistarts (used in opt = "first_order")
		"""

		self.x = x
		self.F = F 
		self.y = F(x)

		self.estimator = estimator

		self.t = 0
		self.beta_mode = beta

		if beta is None:
			self.beta = torch.Tensor([2.]).double()
		else:
			self.beta = beta

		self.epsilon = epsilon
		self.opt = opt
		self.verbose = verbose
		self.multistart = multistart
		self.delta = delta

		self.best_estimate = [None,-10e10]

		self.load_data(self.x,self.y)
		self.fit_model()


	def load_data(self,x,y):
		self.estimator.load_data((x,y))

	def fit_model(self):

		""" Fit a probabilisitc model
		:param x: tensor
		:param y:
		:param iterative: boolean if we want to use Iterative
		:return:
		"""
		self.estimator.fit()
		return None

	def optimize(self):
		"""
		Optimizes current acqusition function
		:return: new point to sample
		"""
		return self.estimator.ucb_optimize(multistart = self.multistart)

	def choose_next(self, xtest):
		"""
		Samples next point according to the acqusition function
		:param xtest:  grid
		:return:
		"""
		if self.opt == "discrete":
			conf = self.estimator.ucb(xtest)
			i = torch.argmax(conf)
			xnext = xtest[i,:].view(-1,1)
			return (xnext,conf[i])

		elif self.opt == "first_order":
			(xnext,val) = self.optimize()
			return (xnext,val)

		else:
			raise NotImplementedError("The requested optimizer is not implemented. ")

	def lcb(self,xtest):
		"""

		:param xtest:
		:return:
		"""

		if self.opt == "first_order":
			xnext, conf_value = self.estimator.ucb_optimize(multistart=self.multistart, lcb = True)
			xnext = xnext
		else:
			conf = self.estimator.lcb(xtest)
			i = torch.argmax(conf)
			conf_value = conf[i]
			xnext = xtest[i].view(1,-1)

		if self.best_estimate[1] < conf_value or self.best_estimate[0] is None:
			self.best_estimate[1] = conf_value
			self.best_estimate[0] = xnext

		return self.best_estimate[0]

	def isin(self,xnext):
		for v in self.x: 
			if np.linalg.norm(v-xnext) < self.epsilon:
				return True
		return False

	def step(self,xtest):
		"""
		Next step of the algorithm
		:param xtest: grid (can be None)
		:return:
		"""
		(xnext,_) = self.choose_next(xtest)
		xnext = xnext.view(-1,1)
		reward = self.F(torch.t(xnext))

		# add to the local database
		self.x = torch.cat((self.x, torch.t(xnext)), dim=0)
		self.y = torch.cat((self.y, reward), dim=0)

		# add to the model database
		self.estimator.add_points((torch.t(xnext), reward.view(1,1)))
		self.fit_model()

		if self.verbose == True:
			print (self.t,torch.t(xnext),reward)
		self.t = self.t + 1

		return (reward,torch.t(xnext))




if __name__=="__main__":

	N = 1
	s = 0.001
	n = 1024
	L_infinity_ball = 0.5
	gamma = 0.1
	d = 1
	BenchmarkFunc = GaussianProcessSample(d = d, n = n, sigma = 0., gamma = gamma, name = "squared_exponential")
	x = BenchmarkFunc.initial_guess(N)

	xtest = BenchmarqqkFunc.interval(n)
	BenchmarkFunc.optimize(xtest, s)
	gamma = BenchmarkFunc.bandwidth()
	bounds = BenchmarkFunc.bounds()
	BenchmarkFunc.scale_max(xtest = xtest)

	F = lambda x: BenchmarkFunc.eval(x,sigma = s)
	m = 256
	embedding = HermiteEmbedding(m = m, gamma = gamma)
	GP = KernelizedFeatures(embedding = embedding, m = m, s = s)
	T = 300
	while True:
		Bandit = UCB_GP(x, F, GP, verbose = True, opt = "general")
		plt.ion()
		for i in range(T):
			Bandit.plot_conf(xtest, d, F =  lambda x: BenchmarkFunc.eval(x,sigma = 0) )
			reward = Bandit.step(xtest)
			plt.pause(0.05)
