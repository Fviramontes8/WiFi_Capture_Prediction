# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:44:11 2021

@author: Frankie
"""
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# Local libraries
import SignalProcessor as sp
import RidgeUtilities as ru
import PlotUtils as pu

"""
TODO: Make this example work
"""

if __name__ == "__main__":
	test_day = "Monday"
	bits_train = np.load("sample_data/tr_bits_15weeks_hoursample_normalized.npy")
	bits_test = np.load("sample_data/tst_bits_week15mon_hoursample_normalized.npy")
	print("Total training data points:", len(bits_train))
	print("Total testing data points:", len(bits_test), '\n')
	#Parameters for formatting training data
	window = 3

	bits_train = sp.buffer(bits_train, window+1, window)
	Xtr = bits_train[:window, :]
	#Xtr = bits_train[0, :]
	Ytr = bits_train[window, :]
	bits_test = sp.buffer(bits_test, window+1, window)
	Xtst = bits_test[:window, :]
	Ytst = bits_test[window, :]
	
	#Transforming the input data so that it can be used in a regressor
	ridge_train_x = Xtr.T
	ridge_train_y = Ytr
	ridge_test_x = Xtst.T
	ridge_test_y = Ytst
	
	print("Training x_shape:", ridge_train_x.shape)
	print("Training y_shape:", ridge_train_y.shape)
	print("Test x_shape:", ridge_test_x.shape)
	print("Test y_shape:", ridge_test_y.shape, '\n')
	
	
	ridge_regressor = ru.linear_ridge(ridge_train_x, ridge_train_y, 1e-4)
	
	ridge_y_pred = ridge_regressor.predict(ridge_test_x)
	print("Ridge prediction shape:", ridge_y_pred.shape)
	pu.plot_ridge_prediction(ridge_y_pred, ridge_test_y, test_day, window)
	
	ridge_self_pred = ridge_regressor.predict(ridge_train_x)
	pu.plot_ridge_prediction(ridge_self_pred, ridge_train_y, test_day, window)
	
	print("Mean squared error from test data prediction:", mse(ridge_test_y, ridge_y_pred))
	print("Mean squared error from training data prediction:", mse(ridge_train_y, ridge_self_pred))
	
	ru.print_ridge_info(ridge_regressor, ridge_train_x, ridge_train_y, ridge_test_x, ridge_test_y)