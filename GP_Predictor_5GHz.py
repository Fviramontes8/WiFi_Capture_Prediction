#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.6
Packages needed: scikit-learn, psycopg2, numpy, scipy
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""
# TODO: Convert this to local libraries
	#Run code with MAPE function, set up a loop for different parameters
#    (and differnt cv folds) to fin best ridge model.

#For matrix and linear algebra calcualtions
import numpy as np

# Scikit learn packages for ML tools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor

# Personal sci kit learn library
import SciKitLearnUtils as sklu

# Personal signal processor/sampler
import SignalProcessor as sp

# Personal plot functions library
import PlotUtils as pu


if __name__ == '__main__':
	test_day = "Monday"
	bits_train = np.load("sample_data/tr_bits_15weeks_hoursample_normalized.npy")
	bits_test = np.load("sample_data/tst_bits_week15mon_hoursample_normalized.npy")
	print(len(bits_train))
	print(len(bits_test))
	#Parameters for formatting training data
	window = 3

	bits_train = sp.buffer(bits_train, window+1, window)
	#print(bits_train.shape)
	Xtr = bits_train[:window, :].T
	#Xtr = bits_train[0, :]
	Ytr = bits_train[window, :]
	bits_test = sp.buffer(bits_test, window+1, window)
	Xtst = bits_test[:window, :].T
	Ytst = bits_test[window, :]
	#print(Xtst.shape)
	
	print("Xtr shape:", Xtr.shape, " Ytr shape:", Ytr.shape)
	print("Xtst shape:", Xtst.shape, "Ytst shape:", Ytst.shape)

	#Declaration of the Gaussian Process Regressor with its kernel parameters
	kernel = sklu.kernel_select("linear")
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                              	normalize_y=False, alpha=1e-3)

	print("Training the Gaussian Process...\n")
	gp.fit(Xtr, Ytr)
	#print("Marginal likelihood:", gp.log_marginal_likelihood())(gp.get_params(deep=True))
	
	#Plotting prediction
	gp_y_pred, gp_y_sigma = gp.predict(Xtst, return_std=True)
	mape_testing_score = sp.mape_test(Ytst, gp_y_pred)
	window_title = str(window)+"\nMAPE: "+str(mape_testing_score)
	pu.plot_gp(gp_y_pred, gp_y_sigma, Ytst, "Time (Halfhours)", "Bits", test_day, window_title)
	
	print("Chi-squared test against real data: ", gp.score(Xtst,Ytst))
	print("MAPE between acutal and estimated:", mape_testing_score)
	
	one_sigma = sklu.verify_one_sigma(Ytst, gp_y_pred, gp_y_sigma)
	two_sigma = sklu.verify_two_sigma(Ytst, gp_y_pred, gp_y_sigma)
	print(one_sigma,"is contained within 1 standard deviation")
	print(two_sigma, "is contained within 2 standard deviations")

	gp_mse = mean_squared_error(Ytst, gp_y_pred)
	print("GP mse: ", gp_mse)