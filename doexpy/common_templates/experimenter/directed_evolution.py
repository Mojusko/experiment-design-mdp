import copy
import random
import numpy as np
import torch
from doexpy.common_templates.experimenter.random_search import RandomSearch

class DirectedEvolution(RandomSearch):

	def __init__(self, F, model, candidate_set, batch_size = 90, xinit = None, yinit = None):
		super().__init__(F, model, candidate_set, batch_size = batch_size, xinit=xinit, yinit=yinit)
		"""
			implements a directed evolution algorithm		
		"""
		dim = self.candidate_set.get_dim()
		elements = self.candidate_set.get_options_per_dim()
		sizes = [1]


		for i in range(dim-1):
			sizes.append(len(elements[i]))
		sizes = np.cumprod(sizes)
		self.convert = lambda x: int(sum([x[i]*sizes[i] for i in range(dim)]))



	def step(self):
		print ("Step:", self.t)
		if self.best_so_far is not None:
			current_base = copy.deepcopy(self.best_so_far).view(1,-1)
		else:
			current_base = self.candidate_set.get_random_elements(size = 1)

		selection_s = []
		selection_x = []

		d = self.candidate_set.get_dim()
		elements = self.candidate_set.get_options_per_dim()

		while len(selection_s) < self.batch_size:


			# shuffle a list called sites
			sites = list(range(d))
			random.shuffle(sites)

			for i in sites:
				for j in elements[i]:
					new_mutation = copy.deepcopy(current_base)
					new_mutation[0,i] = j
					signature = np.apply_along_axis(self.convert, 1, new_mutation)
					if signature not in self.already_sampled and signature not in selection_s:
						selection_s.append(signature)
						selection_x.append(new_mutation)

			# exhausted one site change - randomly mutate another
			k = np.random.randint(d)
			m = np.random.randint(len(elements[k]))
			current_base[0,k] = elements[k][m]

		self.already_sampled = self.already_sampled + selection_s

		xnext = torch.vstack(selection_x)
		ynext = self.F(xnext)

		self.update(xnext,ynext)

		return xnext, ynext