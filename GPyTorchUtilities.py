# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:58:52 2020

@author: Frankie
"""

import torch
import gpytorch


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
		# Must be loss.backward() if there is a single value for loss
		# Otherwise should be loss.sum().backward() or loss.mean().backward() for multiple values for loss
		#loss.sum().backward()
		loss.mean().backward()
		
		print("Iter %02d/%d - Loss: %.3f\tnoise: %.3f" % (
			i + 1, TrainingIter, loss.mean().item(),
			GPModel.likelihood.noise.item()
		))
		

		GPOptimizer.step()
	return GPModel, GPLikelihood

def TorchTest(Xtst, GPModel, GPLikelihood):
	GPModel.eval()
	GPLikelihood.eval()
	
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		output = GPModel(Xtst)
		observed_pred = GPLikelihood(output)

	return observed_pred

"""
Takes a torch tensor and returns lower and upper confidence region for one
standard deviation
"""
def ToStdDev1(pred_mean):
	lower2sigma, upper2sigma = pred_mean.confidence_region()

	lower1sigma = ( (lower2sigma.numpy() - pred_mean.mean.numpy()) / 1.96) + pred_mean.mean.numpy()
	upper1sigma = ( (upper2sigma.numpy() - pred_mean.mean.numpy()) / 1.96) + pred_mean.mean.numpy()

	return lower1sigma, upper1sigma
		
"""
Returns the percent of data contained with in 1 and 2 standard deviations
YPred should be a torch tensor where YPred.mean.numpy() is a valid method call
YTrue should be a numpy array
"""
def verify_confidence_region(YPred, YTrue):
	y_pred_mean = YPred.mean.numpy()
	y_true = YTrue.numpy()
	assert (len(y_pred_mean) == len(YTrue))

	y_lower_sigma, y_upper_sigma = YPred.confidence_region()
	y_lower_sigma1, y_upper_sigma1 = ToStdDev1(YPred)
	y_lower_sigma2, y_upper_sigma2 = y_lower_sigma.numpy(), y_upper_sigma.numpy()

	sigma1_count = 0
	sigma2_count = 0

	for i in range(len(y_pred_mean)):
		if y_lower_sigma1[i] <= y_true[i] and y_upper_sigma1[i] >= y_true[i]:
			sigma1_count += 1

		if y_lower_sigma2[i] <= y_true[i] and y_upper_sigma2[i] >= y_true[i]:
			sigma2_count += 1

	return sigma1_count/len(YTrue), sigma2_count/len(YTrue)
