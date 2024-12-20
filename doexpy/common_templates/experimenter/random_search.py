import numpy as np
import torch


class RandomSearch():

	def __init__(self, F, estimator, candidate_set, replacement = False, batch_size = 90, xinit = None, yinit = None):
		self.candidate_set = candidate_set
		self.batch_size = batch_size
		self.already_sampled = []
		self.replacement = replacement
		self.F = F
		self.estimator = estimator

		if xinit is not None:
			self.x_seen = xinit
			self.y_seen = yinit
			self.best_so_far = self.x_seen[torch.argmax(self.y_seen),:]
			self.best_so_far_value = torch.max(self.y_seen)
		else:
			self.x_seen = None
			self.y_seen = None
			self.best_so_far = None
			self.best_so_far_value = None
		self.t = 1

	def suggest(self, size = 1):
		indices = torch.topk(self.y_seen.view(-1),k = size)[1]
		return self.x_seen[indices], self.y_seen[indices]

	def update(self,xnext,ynext):

		if self.best_so_far is None:
			index = torch.argmax(ynext)
			self.best_so_far_value = torch.max(ynext)
			self.best_so_far = xnext[index]
		else:
			index = torch.argmax(ynext)
			value = torch.max(ynext)
			if value > self.best_so_far_value:
				self.best_so_far_value = value
				self.best_so_far = xnext[index]

		if self.x_seen is None:
			self.x_seen = xnext
			self.y_seen = ynext
		else:
			self.x_seen = torch.vstack([self.x_seen,xnext])
			self.y_seen = torch.vstack([self.y_seen,ynext])

		self.t = self.t + 1


	def step(self):
		print ("Step:",self.t, "best:", self.best_so_far_value)
		if self.replacement:
			selection = self.step_replacement()
		else:
			selection = self.step_without_replacement()

		x = self.candidate_set.get_options()
		xnext = x[selection,:]
		ynext = self.F(xnext)

		self.already_sampled = self.already_sampled + selection
		self.update(xnext,ynext)
		
		if self.estimator is not None:
			self.estimator.add_points((xnext, ynext.view(1, 1)))
			self.estimator.fit()

		return xnext, ynext

	def predict_best_variant(self, batch_size):
		x = None
		y_predicted = None
		y_actual = None
		return x,y_predicted, y_actual

	def step_replacement(self):
		n = self.candidate_set.get_set_size()
		indices = np.arange(0,n,1)
		selection = np.random.choice(indices, self.batch_size).tolist()
		return selection

	def step_without_replacement(self):
		n = self.candidate_set.get_set_size()
		indices = np.arange(0,n,1).tolist()
		if self.already_sampled is not None:
			new_indices = [i for i in indices if i not in self.already_sampled]
		else:
			new_indices = indices
		selection = np.random.choice(new_indices, self.batch_size).tolist()
		return selection




