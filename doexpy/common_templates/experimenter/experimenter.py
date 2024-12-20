import torch

class Experimenter():

	def __init__(self, xinit = None, yinit = None):
		self.already_sampled = []
		self.t = 1


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