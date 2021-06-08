# -*- coding: utf-8 -*-
"""
Python 3.8
Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""

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
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

from scipy.stats import ttest_ind

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
	bits_train_raw = np.load("data/tr_bits_15weeks_hoursample.npy")
	bits_train, tr_mu, tr_sig = sp.std_normalization(bits_train_raw, 1)
	bits_test_raw = np.load("data/tst_bits_week15mon_hoursample.npy")
	bits_test, tst_mu, tst_sig = sp.std_normalization(bits_test_raw, 1)
	print(len(bits_train))
	print(len(bits_test))
	#Parameters for formatting training data
	window = 3

	bits_train = sp.buffer(bits_train, window+1, window)
	#print(bits_train.shape)
	Xtr = bits_train[:window, :]
	#Xtr = bits_train[0, :]
	Ytr = bits_train[window, :]
	bits_test = sp.buffer(bits_test, window+1, window)
	#print(bits_test.shape)
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
	#print("Xtst shape:", Xtst_torch.shape, "Ytst shape:", Ytst.shape)
	
	torch_pred = gptu.TorchTest(Xtst_torch, model, likelihood)
	#print("Prediction shape:", torch_pred.mean.shape)
	
	# Setting up data to print the predition results
	time_test = torch.Tensor([i for i in range(Ytst.shape[0])])
	
	gp_title = "Linear GP Prediction\nwith 1 standard deviation\nand 2 standard deviations"
	x_title = "Time (half-hours)"
	y_title = "Bits"
	pu.PlotGPPred(time_test, torch.Tensor(Ytst), time_test, torch_pred, x_title, y_title, gp_title)
	
	lower_sigma2, upper_sigma2 = torch_pred.confidence_region()
	lower_sigma1, upper_sigma1 = gptu.ToStdDev1(torch_pred)
	
	one_sigma, two_sigma = gptu.verify_confidence_region(torch_pred, torch.Tensor(Ytst))
	print(one_sigma,"is contained within 1 standard deviation")
	print(two_sigma, "is contained within 2 standard deviations\n")
	
	
	y_tst_denorm = sp.denormalize(Ytst, tr_mu, tr_sig)
	t_pred_denorm = sp.denormalize(torch_pred.mean.numpy(), tr_mu, tr_sig)
	title = "Comparison of denormalized data"
	xtitle = "Time (hours)"
	ytitle = "Denormalized bits"
	pu.general_double_plot(y_tst_denorm, t_pred_denorm, title, xtitle, ytitle)
	
	# Lower the better
	torch_gp_mape = sp.mape_test(y_tst_denorm, t_pred_denorm)
	print("Torch mape:", torch_gp_mape)
	
	# Lower the better
	torch_gp_mse = mse(y_tst_denorm, t_pred_denorm)
	print("Torch GP MSE:", torch_gp_mse)
	
	# Lower the better
	torch_gp_mae = mae(y_tst_denorm, t_pred_denorm)
	print("Torch mae:", torch_gp_mae)
	
	torch_gp_r_sq = r2_score(y_tst_denorm, t_pred_denorm)
	print("Torch r^2 value:", torch_gp_r_sq)
	
	torch_gp_ttest = ttest_ind(y_tst_denorm, t_pred_denorm)
	print("Torch ttest value:", torch_gp_ttest[0])
	print("Torch ttest p-value:", torch_gp_ttest[1])
	
	print("-"*40)
	
	# Normalized
	mape_normalized = sp.mape_test(Ytst, torch_pred.mean.numpy())
	print("Torch mape:", mape_normalized)
	
	mse_normalized = mse(Ytst, torch_pred.mean.numpy())
	print("Torch GP MSE:", mse_normalized)
	
	mae_normalized = mae(Ytst, torch_pred.mean.numpy())
	print("Torch mae:", mae_normalized)
	
	r_sq_normalized = r2_score(Ytst, torch_pred.mean.numpy())
	print("Torch r^2 value:", r_sq_normalized)
	
	ttest_normalized = ttest_ind(Ytst, torch_pred.mean.numpy())
	print("Torch ttest value:", ttest_normalized[0])
	print("Torch ttest p-value:", ttest_normalized[1])
	