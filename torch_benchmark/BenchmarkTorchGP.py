# -*- coding: utf-8 -*-
"""
@author: Frankie
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
import os

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
		
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
	
def TorchTrain(Xtr, Ytr, GPModel, GPLikelihood, GPOptimizer, TrainingIter):
	GPModel.train()
	GPLikelihood.train()
	
	marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
		GPLikelihood, 
		GPModel
	)
	
	for i in range(TrainingIter):
		GPOptimizer.zero_grad()
		
		output = GPModel(Xtr)
		
		loss = -marginal_log_likelihood(output, Ytr)
		loss.backward()
		
		#print("Iter", i + 1, "/", TrainingIter)
		GPOptimizer.step()
		
	return model, likelihood

def TorchTest(Xtst, GPModel, GPLikelihood):
	model.eval()
	likelihood.eval()
	
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		observed_pred = GPLikelihood(GPModel(Xtst))
		
	return observed_pred

def PlotGPPred(XCompare, YCompare, XPred, YPred, Title=""):
	with torch.no_grad():
		fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		
		lower_sigma, upper_sigma = YPred.confidence_region()
		ax.plot(XCompare.numpy(), YCompare.numpy(), "k.")
		ax.plot(XPred.numpy(), YPred.mean.numpy(), "b")
		ax.fill_between(xtst.numpy(), lower_sigma.numpy(), upper_sigma.numpy(), alpha = 0.5)
		ax.set_ylim([-10, 10])
		ax.set_xlim([-6, 6])
		ax.set_title(Title)
		ax.legend(["Observed Data", "Mean", "Confidence"])

if __name__ == "__main__":	
	xtr = torch.linspace(-2, 2, 10)
	ytr = xtr + torch.randn(len(xtr))
	
	xtst = torch.linspace(-6, 6, 200)
	ytst = xtst + torch.randn(len(xtst))
	
	
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(xtr, ytr, likelihood)
	
	smoke_test = ("CI" in os.environ)
	training_iter = 2 if smoke_test else 50
	
	optimizer = torch.optim.Adam([
			{"params" : model.parameters()},
		],
		lr = 0.1
	)
	
	model, likelihood = TorchTrain(xtr, ytr, model, likelihood, optimizer, training_iter)	
	pred = TorchTest(xtst, model, likelihood)
		
	PlotGPPred(xtr, ytr, xtst, pred)
	PlotGPPred(xtst, ytst, xtst, pred)
	
	xtr2 = torch.linspace(-6, 6, 10)
	ytr2 = xtr2 + torch.randn(len(xtr2))
	xtst2 = torch.linspace(-6, 6, 200)
	ytst2 = xtst2 + torch.randn(len(xtst2))
	
	model = ExactGPModel(xtr2, ytr2, likelihood)
	
	model, likelihood = TorchTrain(xtr2, ytr2, model, likelihood, optimizer, training_iter)	
	pred = TorchTest(xtst2, model, likelihood)

	"""
	l, u = pred.confidence_region()
	print(len(pred.mean.numpy()), len(l.numpy()), len(u.numpy()))
	z = []
	np_pred = pred.mean.numpy()
	for i in range(len(np_pred)):
		if np_pred[i] > -2 and np_pred[i] < 2:
			z.append(np_pred[i])
		
	print(len(z))
	"""
	
	PlotGPPred(xtr2, ytr2, xtst2, pred)
	PlotGPPred(xtst2, ytst2, xtst2, pred)