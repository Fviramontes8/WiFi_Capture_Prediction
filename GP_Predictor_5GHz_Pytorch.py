# -*- coding: utf-8 -*-
"""
Python 3.8
Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""
# TODO: 
#	Try batch GP
#	Run code with MAPE function, set up a loop for different parameters
#    (and differnt cv folds) to find best ridge model.

# Private signal processor/sampler
import SignalProcessor as sp
# Private GPyTorch and PyTorch functions
import GPyTorchUtilities as gptu
# Private plot functions
import PlotUtils as pu

# For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

# Mean squared error to determine quality of prediction
from sklearn.metrics import mean_squared_error as mse

import torch
import gpytorch

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

if __name__ == "__main__":
	test_day = "Monday"
	bits_train = np.load("data/tr_bits_15weeks_hoursample_normalized.npy")
	bits_test = np.load("data/tst_bits_week15mon_hoursample_normalized.npy")
	print(len(bits_train))
	print(len(bits_test))
	#Parameters for formatting training data
	window = 3

	bits_train = sp.buffer(bits_train, window+1, window)
	print(bits_train.shape)
	Xtr = bits_train[:window, :]
	#Xtr = bits_train[0, :]
	Ytr = bits_train[window, :]
	bits_test = sp.buffer(bits_test, window+1, window)
	print(bits_test.shape)
	Xtst = bits_test[:window, :]
	Ytst = bits_test[window, :]
	
	# Begin Pytorch training
	# Need to cast dtype as torch.double (or torch.float64)
	Xtr_torch, Ytr_torch = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
	Xtr_torch = Xtr_torch.transpose(0, 1)
	print("Xtr shape:", Xtr_torch.shape, "Ytr shape:", Ytr_torch.shape)
	
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(Xtr_torch, Ytr_torch, likelihood)
	
	optimizer = torch.optim.Adam([
			{"params" : model.parameters()},
		],
		lr = 0.1
	)
	
	gptu.TorchTrain(Xtr_torch, Ytr_torch, model, likelihood, optimizer, 100)
	
	Xtst_torch = torch.from_numpy(Xtst).float()
	Xtst_torch = Xtst_torch.transpose(0, 1)
	print("Xtst shape:", Xtst_torch.shape, "Ytst shape:", Ytst.shape)
	
	torch_pred = gptu.TorchTest(Xtst_torch, model, likelihood)
	print("Prediction shape:", torch_pred.mean.shape)
	
	# Setting up data to print the predition results
	time_test = torch.Tensor([i for i in range(Ytst.shape[0])])
	
	gp_title = "Linear GP Prediction\nwith 1 standard deviation\nand 2 standard deviations"
	x_title = "Time (hours)"
	y_title = "Bits"
	pu.PlotGPPred(time_test, torch.Tensor(Ytst), time_test, torch_pred, x_title, y_title, gp_title)
	
	torch_gp_mape = sp.mape_test(Ytst, torch_pred.mean.numpy())
	print("Torch mape:", torch_gp_mape)
	torch_gp_mse = mse(Ytst, torch_pred.mean.numpy())
	print("Torch GP MSE:", torch_gp_mse)
	
	
	lower_sigma2, upper_sigma2 = torch_pred.confidence_region()
	lower_sigma1, upper_sigma1 = gptu.ToStdDev1(torch_pred)
	
	one_sigma, two_sigma = gptu.verify_confidence_region(torch_pred, torch.Tensor(Ytst))
	print(one_sigma,"is contained within 1 standard deviation")
	print(two_sigma, "is contained within 2 standard deviations")