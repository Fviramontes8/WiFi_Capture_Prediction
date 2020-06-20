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
		self.covar_module = gpytorch.kernels.ScaleKernel(
			gpytorch.kernels.LinearKernel()
		)
		
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
		ax.fill_between(
			xtst.numpy(), 
			lower_sigma.numpy(), 
			upper_sigma.numpy(), 
			alpha = 0.5
		)
		ax.set_ylim([-10, 10])
		ax.set_xlim([-6, 6])
		ax.set_title(Title)
		ax.legend(["Observed Data", "Mean", "Confidence"])
		
def verify_confidene_region(YPred, YTrue):
	y_pred_mean = YPred.mean.numpy()
	y_true = YTrue.numpy()
	assert (len(y_pred_mean) == len(YTrue))
	
	y_lower_sigma, y_upper_sigma = YPred.confidence_region()
	y_lower_sigma1, y_upper_sigma1 = y_lower_sigma.numpy() / 1.96, y_upper_sigma.numpy() / 1.96
	y_lower_sigma2, y_upper_sigma2 = y_lower_sigma.numpy(), y_upper_sigma.numpy()
	
	sigma1_count = 0
	sigma2_count = 0
	
	for i in range(len(y_pred_mean)):
		if y_lower_sigma1[i] <= y_true[i] and y_upper_sigma1[i] >= y_true[i]:
			sigma1_count += 1
		
		if y_lower_sigma2[i] <= y_true[i] and y_upper_sigma2[i] >= y_true[i]:
			sigma2_count += 1
			
	return sigma1_count/len(YTrue), sigma2_count/len(YTrue)
	

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
	
	model, likelihood = TorchTrain(
		xtr, 
		ytr, 
		model, 
		likelihood, 
		optimizer, 
		training_iter
	)	
	pred = TorchTest(xtst, model, likelihood)
	print(type(pred))
		
	gp_title = "Linear GP Prediction\nwith 2 standard deviations"
	PlotGPPred(xtr, ytr, xtst, pred, gp_title)
	PlotGPPred(xtst, ytst, xtst, pred, gp_title)
	
	print("1 standard deviation/2 standard deviation confidence:", verify_confidene_region(pred, ytst))
	
	xtr2 = torch.linspace(-6, 6, 10)
	ytr2 = xtr2 + torch.randn(len(xtr2))
	xtst2 = torch.linspace(-6, 6, 200)
	ytst2 = xtst2 + torch.randn(len(xtst2))
	
	model = ExactGPModel(xtr2, ytr2, likelihood)
	
	model, likelihood = TorchTrain(
		xtr2, 
		ytr2, 
		model, 
		likelihood, 
		optimizer, 
		training_iter
	)	
	pred = TorchTest(xtst2, model, likelihood)

	
	PlotGPPred(xtr2, ytr2, xtst2, pred, gp_title)
	PlotGPPred(xtst2, ytst2, xtst2, pred, gp_title)
	
	print("1 standard deviation/2 standard deviation confidence:", verify_confidene_region(pred, ytst2))