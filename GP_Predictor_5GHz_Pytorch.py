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
# Private plot functions
import PlotUtils as pu

# For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

# For ploting data
import matplotlib.pyplot as plt

# Machine Learning package for Ridge regression
from sklearn.linear_model import RidgeCV
# Mean squared error to determine quality of prediction
from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split

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

def tr_data_prep(training, testing, window, validating=0):
	'''Inputs: train/test, which needs to be an array and each can be a different size, window,
		which specifies how wide the resultant matrix of this function is.
	Output: Training and test matricies that has window of the given input values (Xtr, Ytr, Xtst),
		and a graph-able array of what the training values look like (ycomp)
	Example: window = 5, length of array input (both train and test are same size in this example) = n
		[x_0 x_1 ... x_4]          [x_5]
		[x_1 x_2 ... x_5]          [x_6]
	x = [x_2 x_3 ... x_6]     y =  [x_7]
		[... ... ... ...]          [...]
		[x_n-5-1... ... x_n-1]     [x_n]
	'''
	x_valid = np.ones((2, 2))
	y_valid = np.ones((2, 2))
	if(validating):
		training, validation = train_test_split(training, test_size = 0.2)
		x_valid = np.atleast_2d([sp.grab_nz(validation, m, n) for m, n in zip(range(validation.shape[0]), range(window, validation.shape[0]))])
		y_valid = np.atleast_2d([[validation[i] for i in range(window, validation.shape[0])]]).T

	Xtr = np.atleast_2d([sp.grab_nz(training, m, n) for m, n in zip(range(training.shape[0]), range(window, training.shape[0]))])
	Ytr = np.atleast_2d([[training[i] for i in range(window, training.shape[0])]]).T
	Xtst = np.atleast_2d([sp.grab_nz(testing, m, n) for m, n in zip(range(testing.shape[0]), range(window, testing.shape[0]))])
	Ycomp = np.atleast_2d([testing[i] for i in range(window, testing.shape[0])]).T

	return Xtr, Ytr, Xtst, Ycomp, x_valid, y_valid

if __name__ == "__main__":
	days_of_week = ["mon",
				 "tues",
				 "wed",
				 "thurs",
				 "fri"
				 ]
	day = "thurs"
	labels_5ghz = ["Number of users",
       				"Bits"
       				]
	begin_week = 2
	end_week = 14
	init_sample_rate = 60
	second_sample_rate = 30
	test_week=15
	total_weeks=13

	#bits_tr = week_data_prep(day, begin_week, end_week, init_sample_rate, second_sample_rate)
	bits_tr = dbp.day_data_prep(days_of_week, total_weeks, init_sample_rate, second_sample_rate)
	bits_title = "Training data of "+str(total_weeks)+" total weeks (mon-fri)"
	bits_xtitle = "Time (10-minute chunks of multiple days)"
	bits_ytitle = "Bits"
	pu.general_plot(bits_tr, bits_title, bits_xtitle, bits_ytitle)
	
	#Normalizing training data
	bits_tr = sp.std_normalization(bits_tr)
	bits_title = "Normailzed t" + bits_title[1:]
	bits_ytitle += " (normalized)"
	pu.general_plot(bits_tr, bits_title, bits_xtitle, bits_ytitle)