# -*- coding: utf-8 -*-
"""
Python 3.8
Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""
# TODO: Run code with MAPE function, set up a loop for different parameters
#    (and differnt cv folds) to find best ridge model.

# Private signal processor/sampler
import SignalProcessor as sp
# Private database processor
import DatabaseProcessor as dbp
# Private GPyTorch and PyTorch functions
import GPyTorchUtilities as gptu

# For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

# For ploting data
import matplotlib.pyplot as plt

# Machine Learning package for Ridge regression
from sklearn.linear_model import RidgeCV
# Mean squared error to determine quality of prediction
from sklearn.metrics import mean_squared_error as mse

import torch
import gpytorch

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

