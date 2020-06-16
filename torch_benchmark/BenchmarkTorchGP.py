# -*- coding: utf-8 -*-
"""
@author: Frankie
"""

import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import os

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
		#self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
		
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":	
	xtr = torch.linspace(-2, 2, 10)
	ytr = xtr + torch.randn(len(xtr))
	plt.plot(xtr, ytr)
	plt.show()
	
	xtst = torch.linspace(-6, 6, 200)
	
	
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(xtr, ytr, likelihood)
	
	smoke_test = ("CI" in os.environ)
	training_iter = 2 if smoke_test else 50
	
	# Get optimal hyperparameters
	model.train()
	likelihood.train()
	
	optimizer = torch.optim.Adam([
			{"params" : model.parameters()},
		],
		lr = 0.4
	)
	
	# Marginal log likelihood
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
	
	for i in range(training_iter):
		# Zero out gradients from previous iteration
		optimizer.zero_grad()
		
		# Output from the model
		output = model(xtr)
		
		# Calculate loss and backprop gradients
		loss = -mll(output, ytr)
		loss.backward()
		
		#print("Iter %02d/%d - Loss: %.3f\tlengthscale: %.3f\tnoise: %.3f" % (
		print("Iter %02d/%d\tnoise: %.3f" % (
			i + 1, training_iter, loss.item(), 
			#model.covar_module.base_kernel.lengthscale.item(),
			model.likelihood.noise.item()
		))
		optimizer.step()
		
	
	model.eval()
	likelihood.eval()
	
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		observed_pred = likelihood(model(xtst))
		
	with torch.no_grad():
		fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		
		lower_sigma, upper_sigma = observed_pred.confidence_region()
		ax.plot(xtr.numpy(), ytr.numpy(), "k-.")
		ax.plot(xtst.numpy(), observed_pred.mean.numpy(), "b")
		ax.fill_between(xtst.numpy(), lower_sigma.numpy(), upper_sigma.numpy(), alpha = 0.5)
		ax.set_ylim([-6, 6])
		ax.set_xlim([-10, 10])
		ax.legend(["Observed Data", "Mean", "Confidence"])
	