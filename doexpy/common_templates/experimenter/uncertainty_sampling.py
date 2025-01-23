import warnings
import copy
import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP

from doexpy.common_templates.experimenter.ucb import UCB

warnings.filterwarnings("ignore",".*GUI is implemented.*")

class UncertaintySampling(UCB):

	def __init__(self,F,model, candidate_set, beta = None,
				 batch_size = 1, diversity = 'none', xinit = None, yinit = None, agg = 'sum', replacement = True):
		super().__init__(F,model, candidate_set,xinit = xinit, yinit = yinit, batch_size=batch_size)
		self.replacement = replacement
		self.agg = agg
		self.selected_utility = []


	def utility(self,xtest):

		if self.agg == 'max':
			(ymean,ystd) = self.model.mean_std(xtest)
			conf = ystd

		elif self.agg == 'sum':
			pass
			embed = self.model.embedding.embed
			n = self.model.x.size()[0]

			phi =  embed(xtest)
			V = phi.T@phi

			#K = kernel(self.model.x, self.model.x) + torch.eye(n).double() * self.model.s**2 * self.model.get_lam()
			V_collection = torch.einsum('ij,ik->ijk',phi,phi)
			V_initial = embed(self.model.x).T@embed(self.model.x) + torch.eye(self.model.m).double() * self.model.s**2 * self.model.get_lam()
			V_initial = V_initial.reshape(1, V_initial.size()[0], V_initial.size()[1])
			V_initial_inverse = torch.inverse(V_initial + V_collection)
			conf = torch.einsum('ijk,kj->i',V_initial_inverse,V)

		if self.replacement == False:
			conf[self.selected_indices] = -np.inf
		else:
			pass
		return conf

		#
		# 	def block_matrix_inverse(A_inv, B, C, D):
		# 		# Compute the Schur complement
		# 		S = D - C @ A_inv @ B
		#
		# 		# Compute the inverse of the Schur complement
		# 		S_inv = torch.inverse(S)
		#
		# 		# Calculate the blocks of the inverse matrix
		# 		top_left = A_inv + A_inv @ B @ S_inv @ C @ A_inv
		# 		top_right = -A_inv @ B @ S_inv
		# 		bottom_left = -S_inv @ C @ A_inv
		# 		bottom_right = S_inv
		#
		# 		# Combine the blocks to form the full inverse matrix
		# 		K_inv = torch.cat([
		# 			torch.cat([top_left, top_right], dim=1),
		# 			torch.cat([bottom_left, bottom_right], dim=1)
		# 		], dim=0)
		#
		# 		return K_inv
		#
		# 	prior_x = self.model.x
		# 	conf = torch.zeros(xtest.size()[0])
		# 	n = self.model.x.size()[0]
		# 	zeros = torch.zeros(size = (n+1,1)).double()
		# 	kernel = self.model.kernel_object.kernel
		# 	Kstarstar = kernel(xtest, xtest)
		# 	K = kernel(prior_x, prior_x) + torch.eye(n).double()*self.model.s
		# 	Kinv = torch.inverse(K)
		# 	for j in range(xtest.size()[0]):
		# 		print ("j",j)
		# 		new_x = torch.cat([prior_x, xtest[j].view(1,-1)], dim = 0)
		# 		b = kernel(prior_x, xtest[j].view(1,-1))
		# 		Kstar = kernel(new_x,xtest)
		# 		c = kernel (xtest[j].view(1,-1), xtest[j].view(1,-1))
		# 		#K = kernel(new_x,new_x) + torch.eye(n+1).double()*self.model.s
		# 		conf[j] = torch.trace(Kstarstar - Kstar.mm(block_matrix_inverse(Kinv,b.T,b,c)).mm(Kstar.T))
		# return conf


	def choose_next(self):
		xtest = self.candidate_set.get_options()
		xoriginal = self.candidate_set.get_options_raw()

		utility = self.utility(xtest).view(-1)
		if self.batch_size > 1:
			# randomly subsample
			ind = torch.topk(utility.view(-1), k = int(min(xtest.size()[0],10*self.batch_size)))[1]
			_, K_opt = self.model.mean_std(xtest[ind,:], full=True)
			jitter = 10e-3
			K_opt = K_opt + torch.eye(n=K_opt.size()[0]).double() * jitter
			DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})
			rng = np.random.RandomState(np.random.randint(100))
			sample = DPP.sample_exact_k_dpp(size=self.batch_size, random_state=rng, model="KuTa12")
			xnext = xtest[ind, :][sample, :]
			xnext_original = xoriginal[ind, :][sample, :]
			self.selected_indices.append(ind[sample])
		else:
			ind = torch.argmax(utility.view(-1))
			xnext = xtest[ind, :].view(1,-1)
			xnext_original = xoriginal[ind, :].view(1,-1)
			self.selected_indices.append(int(ind))
			self.selected_utility.append(utility[ind])
		return xnext, xnext_original