# -*- coding: utf-8 -*-
"""
@author: Frankie
"""

import torch
import gpytorch
from sklearn.metrics import mean_squared_error as mse
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
			xtst.numpy(),
			lower_sigma.numpy(),
			upper_sigma.numpy(),
			alpha = 0.5
		)

		ax.fill_between(
			xtst.numpy(),
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


if __name__ == "__main__":
	#########################################################
	# First experiment
	#########################################################

	# Generating training data
	xtr = torch.linspace(-2, 2, 10)
	ytr = xtr + torch.randn(len(xtr))

	# Generating testing data
	xtst = torch.linspace(-6, 6, 200)
	ytst = xtst + torch.randn(len(xtst))

	# Likelihood and Gaussian Process (GP) model
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(xtr, ytr, likelihood)

	# Determining training iterations
	smoke_test = ("CI" in os.environ)
	training_iter = 2 if smoke_test else 50

	# Optimizer for GP
	optimizer = torch.optim.Adam([
			{"params" : model.parameters()},
		],
		lr = 0.1
	)

	# Training the model
	model, likelihood = TorchTrain(
		xtr,
		ytr,
		model,
		likelihood,
		optimizer,
		training_iter
	)

	# Making a prediction on the test data set
	pred = TorchTest(xtst, model, likelihood)

	# Plotting GP results on training data
	gp_title = "Linear GP Prediction\nwith 1 standard deviation confidence\nand 2 standard deviation confidence"
	PlotGPPred(xtr, ytr, xtst, pred, gp_title)

	# Extraction confidence regions to include to title of the plot for testing data
	tst_sigma1, tst_sigma2 = verify_confidence_region(pred, ytst)
	gp_title = "Linear GP Prediction\nwith 1 standard deviation confidence (" + \
		str(tst_sigma1) + ")\nand 2 standard deviation confidence (" + \
		str(tst_sigma2) + ")"

	# Plotting GP results on testing data
	PlotGPPred(xtst, ytst, xtst, pred, gp_title)

	print("1 standard deviation/2 standard deviation confidence for test:", tst_sigma1, tst_sigma2)
	print("MSE:", mse(ytst, pred.mean.numpy()))


	#########################################################
	# Second experiment
	#########################################################

	# Generating second set of training data
	xtr2 = torch.linspace(-6, 6, 10)
	ytr2 = xtr2 + torch.randn(len(xtr2))

	# Generating second set of testing data
	xtst2 = torch.linspace(-6, 6, 200)
	ytst2 = xtst2 + torch.randn(len(xtst2))

	# Crating new GP model on new training data
	model = ExactGPModel(xtr2, ytr2, likelihood)

	# Training new GP model
	model, likelihood = TorchTrain(
		xtr2,
		ytr2,
		model,
		likelihood,
		optimizer,
		training_iter
	)

	# Making a prediction on the new testing data set
	pred = TorchTest(xtst2, model, likelihood)

	# Ploting GP results on training data set
	gp_title = "Linear GP Prediction\nwith 1 standard deviation confidence\nand 2 standard deviation confidence"
	PlotGPPred(xtr2, ytr2, xtst2, pred, gp_title)

	# Extracting confidence region and adding them to title for test data set
	tst2_sigma1, tst2_sigma2 = verify_confidence_region(pred, ytst2)
	gp_title = "Linear GP Prediction\nwith 1 standard deviation confidence (" + \
		str(tst2_sigma1) + ")\nand 2 standard deviation confidence (" + \
		str(tst2_sigma2) + ")"

	# Plotting GP results on testing data set
	PlotGPPred(xtst2, ytst2, xtst2, pred, gp_title)

	print("1 standard deviation/2 standard deviation confidence:", tst2_sigma1, tst2_sigma2)
	print("MSE", mse(ytst2, pred.mean.numpy()))