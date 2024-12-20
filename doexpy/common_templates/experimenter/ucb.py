import warnings

from doexpy.common_templates.experimenter.experimenter import Experimenter

warnings.filterwarnings("ignore",".*GUI is implemented.*")
import torch
import numpy as np
from dppy.finite_dpps import FiniteDPP

class UCB(Experimenter):
	def __init__(self,F,model, candidate_set, beta = None,
				 batch_size = 1, diversity = 'none', xinit = None, yinit = None, verbose = False):
		super().__init__(xinit = xinit, yinit = yinit)
		self.candidate_set = candidate_set
		self.F = F
		self.model = model
		self.verbose = verbose

		self.batch_size = batch_size
		self.diversity = diversity
		self.selected_indices = []
		if beta is None:
			self.beta = torch.Tensor([2.]).double()
		else:
			self.beta = beta

	def utility(self,xtest):
		(ymean,ystd) = self.model.mean_std(xtest)
		conf = ymean + torch.sqrt(self.beta)*ystd
		return conf

	def get_selected(self):
		return self.selected_indices

	def choose_next(self):
		xtest = self.candidate_set.get_options()
		xoriginal = self.candidate_set.get_options_raw()
		utility = self.utility(xtest).view(-1)

		if self.diversity == 'none':
			ind = torch.topk(utility, k = self.batch_size)[1]
			xnext = xtest[ind,:]
			xnext_original = xoriginal[ind,:]
			self.selected_indices.append(np.arange(0,xtest.size()[0],1)[ind])

		elif self.diversity == 'dpp':
			rng = np.random.RandomState(np.random.randint(100))
			ind = torch.topk(utility.view(-1), k=int(min(self.batch_size * 4,xtest.size()[0])))[1]
			#_, K_opt = self.model.mean_std(xtest[ind, :], full=True)
			K_opt = self.model.model_similarity(xtest[ind,:])
			jitter = 10e-3
			K_opt = K_opt + torch.eye(n = K_opt.size()[0]).double()*jitter
			DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})
			sample = DPP.sample_exact_k_dpp(size=self.batch_size,
											random_state=rng, model="KuTa12")
			xnext = xtest[ind, :][sample, :]
			xnext_original = xoriginal[ind,:][sample, :]
			self.selected_indices.append(ind[sample])

		return xnext, xnext_original

	def suggest(self, size = 1):
		xtest = self.candidate_set.get_options()
		xoriginal = self.candidate_set.get_options_raw()
		ymean,ystd = self.model.mean_std(xtest)
		utility = ymean - torch.sqrt(self.beta)*ystd

		if self.diversity == 'non':
			ind = torch.topk(utility.view(-1), k = size)[1]
			xnext = xtest[ind,:]
			xnext_original = xoriginal[ind,:]

		elif self.diversity == 'dpp':
			rng = np.random.RandomState(np.random.randint(100))
			ind = torch.topk(utility.view(-1), k = size*4)[1]
			_, K_opt = self.model.mean_std(xtest[ind,:], full=True)
			jitter = 10e-3
			K_opt = K_opt + torch.eye(n = K_opt.size()[0]).double()*jitter
			DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})
			sample = DPP.sample_exact_k_dpp(size=size, random_state=rng, model="KuTa12")
			xnext = xtest[ind,:][sample,:]
			xnext_original = xoriginal[ind,:][sample,:]

		suggestion_x = xnext

		suggestion_values = self.F(xnext_original)
		#print (ymean[ind].T)
		#print (suggestion_values.T)

		return suggestion_x, suggestion_values

	def step(self):
		if self.verbose:
			print ("Step:", self.t, self.best_so_far_value)
		xnext,xnext_original = self.choose_next()
		ynext = self.F(xnext_original)

		# add to the model database
		self.model.add_points((xnext, ynext))

		# update counters
		self.update(xnext,ynext)

		return (xnext,ynext)



if __name__=="__main__":
	pass