import copy
import warnings

import torch

from doexpy.optimization.ucb import UCB

warnings.filterwarnings("ignore",".*GUI is implemented.*")

class ExploreRemove(UCB):

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
		mask = utility>0
		xtest_selected = xtest[mask]
		xoriginal_selected = xoriginal[mask]
		new_model = copy.deepcopy(self.model)
		selection = []
		selection_original = []
		# print("How many?:",self.model.x.size())
		# print("Variance sampling")
		for i in range(self.batch_size):
			(ymean, ystd) = new_model.mean_std(xtest_selected)
			index = torch.argmax(ystd)
			xs =  xtest_selected[index,:].view(1,-1)
			xs_original =xoriginal_selected[index,:].view(1,-1)
			selection.append(xs)
			selection_original.append(xs_original)
			ys = ymean[index,:].view(1,1)
			new_model.add_data_point(xs,ys)

		xnext = torch.vstack(selection)
		xnext_original  = torch.vstack(selection_original)
		# print ("new model:",new_model.x.size())
		# print ("old model:",self.model.x.size())
		return xnext, xnext_original
