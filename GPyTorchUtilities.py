# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:58:52 2020

@author: Frankie
"""

import torch
import gpytorch

import numpy as np
import matplotlib.pyplot as plt

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

	return GPModel, GPLikelihood

def TorchTest(Xtst, GPModel, GPLikelihood):
	GPModel.eval()
	GPLikelihood.eval()

	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		observed_pred = GPLikelihood(GPModel(Xtst))

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

def PlotGPPred(XCompare, YCompare, XPred, YPred, Title=""):
	with torch.no_grad():
		fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		lower_sigma, upper_sigma = YPred.confidence_region()

		sigma1_lower, sigma1_upper = ToStdDev1(YPred)

		ax.plot(XCompare.numpy(), YCompare.numpy(), "k.")
		ax.plot(XPred.numpy(), YPred.mean.numpy(), "b")
		ax.fill_between(
			XPred.numpy(),
			lower_sigma.numpy(),
			upper_sigma.numpy(),
			alpha = 0.5
		)

		ax.fill_between(
			XPred.numpy(),
			sigma1_lower,
			sigma1_upper,
			alpha = 0.5
		)

		ax.set_ylim([-10, 10])
		ax.set_xlim([-6, 6])
		ax.set_title(Title)
		ax.legend(["Observed Data", "Prediction Mean", "2 StdDev Confidence", "1 StdDev confidence"])
		
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

def mape_test(actual, estimated):
	assert(type(actual) == np.ndarray)
	assert(type(estimated) == np.ndarray)
	"""
	size = len(actual)
	result = 0.0
	for i in range(size):
		result += np.abs( (actual[i] - estimated[i])  / actual[i] )
	result /= size
	"""
	result = np.mean(np.abs((actual-estimated)/actual))
	return float(result)