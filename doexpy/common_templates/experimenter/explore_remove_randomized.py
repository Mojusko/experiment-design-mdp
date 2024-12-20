import warnings

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP

from doexpy.optimization.ucb import UCB

warnings.filterwarnings("ignore",".*GUI is implemented.*")

class ExploreRemoveRandomized(UCB):

	def utility(self,xtest):
		(ymean,ystd) = self.model.mean_std(xtest)
		conf = ystd
		maxlcb = torch.max(ymean - torch.sqrt(self.beta)*ystd)
		ucb = ymean + torch.sqrt(self.beta)*ystd
		conf[ucb < maxlcb] = -1000.
		return conf

	def choose_next(self):
		xtest = self.candidate_set.get_options()
		xoriginal = self.candidate_set.get_options_raw()

		utility = self.utility(xtest).view(-1)
		print (utility)
		mask = utility>0
		xtest_selected = xtest[mask]
		xoriginal_selected = xoriginal[mask]

		(ymean, ystd) = self.model.mean_std(xtest_selected)

		# randomly subsample
		ind = torch.topk(ystd.view(-1), k = min(10*self.batch_size,ystd.size()[0]))[1]
		print (ind)
		_, K_opt = self.model.mean_std(xtest_selected[ind,:], full=True)
		jitter = 10e-2
		K_opt = K_opt + torch.eye(n=K_opt.size()[0]).double() * jitter
		DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})
		rng = np.random.RandomState(np.random.randint(100))
		sample = DPP.sample_exact_k_dpp(size=self.batch_size, random_state=rng, model="KuTa12")

		xnext = xtest_selected[ind, :][sample, :]
		xnext_original = xoriginal_selected[ind, :][sample, :]

		return xnext, xnext_original
