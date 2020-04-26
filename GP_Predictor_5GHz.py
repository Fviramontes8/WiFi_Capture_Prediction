#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.6
Packages needed: scikit-learn, psycopg2, numpy, scipy
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""
# TODO: Run code with MAPE function, set up a loop for different parameters
#    (and differnt cv folds) to fin best ridge model.
#Private signal processor/sampler
import SignalProcessor as sp

#Private database processor
import DatabaseProcessor as dbp

#For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

#For ploting data
import matplotlib.pyplot as plt

#Scikit learn packages for ML tools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#Machine Learning package for Ridge regression
from sklearn.linear_model import RidgeCV

#Machine Learning package for the Gaussian Process Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ, RBF



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

def plot_gp(pred, sigma, compare, feature, day, window):
	#print("Arguement size: ", pred.shape, sigma.shape, compare.shape)
	#print("Feature: ", feature, "\nDay: ", day, "\nTitle ", window)
	sigma_coef = 0.98#1.96
	prediction_time= [p+1 for p in range(len(pred))]
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(prediction_time, compare, "y-", label="Actual data")
	plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
			  np.concatenate([pred-sigma_coef*sigma,
					 (pred+sigma_coef*sigma)[::-1]]),
			  alpha=.5, fc='b', ec='none')
	plt.legend()
	plt.title("Gaussian Process Prediction with 6th order Butterworth filtering,\nPredicting "
		   +day+"day\nWith window of "+str(window)+"\nAnd sigma of "+str(sigma_coef))
	plt.xlabel("Time (Hours)")
	plt.ylabel(feature+" (predicted)")
	plt.show()

	sigma_coef *= 2
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(prediction_time, compare, "y-", label="Validation data")
	plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
			  np.concatenate([pred-sigma_coef*sigma,
					 (pred+sigma_coef*sigma)[::-1]]),
			  alpha=.5, fc='b', ec='none')
	plt.legend()
	plt.title("Gaussian Process Prediction with 6th order Butterworth filter,\nPredicting "
		   +day+"day\nWith window of "+str(window)+"\nAnd sigma of "+str(sigma_coef))
	plt.xlabel("Time (Hours)")
	plt.ylabel(feature+" (predicted)")
	plt.show()

def plot_ridge(pred, compare, feature, day, window):
	prediction_time= [p+1 for p in range(len(pred))]

	plt.plot(prediction_time, pred, "c-", label="Ridge Regression Prediction")
	plt.plot(prediction_time, compare, "y-", label="Actual data")
	plt.legend()
	plt.title("Ridge Regression Prediction with 6th order Butterworth filtering,\nPredicting "
		   +day+"day\nWith window of "+str(window))
	plt.xlabel("Time (hours)")
	plt.ylabel(feature+" (predicted)")
	plt.show()


def mape_test(actual, estimated):
	if( (type(actual) is np.ndarray) & (type(estimated) is np.ndarray)):
		pass
	else:
		print("Inputs must be data type: numpy.ndarray")
		return -1;

	size = len(actual)
	result = 0.0
	for i in range(size):
		result += float( np.abs(actual[i]-estimated[i]) / actual[i])
		#print("Test resul:", result)
	result /= size
	return result

def kernel_select(kernel_str):
	if type(kernel_str) is not str:
		print("Input is not a string! Defaulting to a linear kernel\n")
	if kernel_str == "linear":
		kernel1 = LK(sigma_0 = 10, sigma_0_bounds=(10e-1, 10e2))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(noise_level=10e0, noise_level_bounds = (10e-4, 10e-2))
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
	elif kernel_str == "RBF":
		kernel1 = RBF(length_scale=10e2, length_scale_bounds=(1e1, 1e3))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(noise_level=1e-2, noise_level_bounds = (10e-3, 10e-1))
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
	else:
		kernel1 = LK(sigma_0 = 10, sigma_0_bounds=(10e-1, 10e1))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(noise_level=1e-4, noise_level_bounds = (10e-5, 10e-1))
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
		print("Not a valid kernel name, defaulting to ", kernel)

	#print(kernel)
	return kernel

def verify_one_sigma(actual_series, pred_series, sigma_series):
	upper_sigma = []
	for i in range(len(pred_series)):
		upper_sigma.append(pred_series[i] + sigma_series[i])

	lower_sigma = []
	for j in range(len(pred_series)):
		lower_sigma.append(pred_series[j] - sigma_series[j])

	sigma_one_count = 0
	for h in range(len(pred_series)):
		if((actual_series[h] <= upper_sigma[h]) & (actual_series[h] >= lower_sigma[h])):
			sigma_one_count += 1

	one_sigma=False
	if( sigma_one_count > (len(actual_series) * 0.65)):
		one_sigma = True

	return one_sigma

def verify_two_sigma(actual_series, pred_series, sigma_series):
	upper_sigma = []
	for i in range(len(pred_series)):
		upper_sigma.append(pred_series[i] + (1.98* sigma_series[i]))

	lower_sigma = []
	for j in range(len(pred_series)):
		lower_sigma.append(pred_series[j] - (1.98* sigma_series[i]))

	sigma_two_count = 0
	for h in range(len(pred_series)):
		if((actual_series[h] <= upper_sigma[h]) & (actual_series[h] >= lower_sigma[h])):
			sigma_two_count += 1

	two_sigma=False

	if( sigma_two_count > (len(actual_series) * 0.95)):
		two_sigma = True

	return two_sigma


if __name__ == '__main__':
	#Reading data from database
	#mon, tues, wed, thurs, fri are full from 2 to 15
	#tues has an interesting result
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
	second_sample_rate = 60
	test_week=15
	total_weeks=13

	#bits_tr = week_data_prep(day, begin_week, end_week, init_sample_rate, second_sample_rate)
	bits_tr = dbp.day_data_prep(days_of_week, total_weeks, init_sample_rate, second_sample_rate)
	plt.plot(bits_tr)
	plt.xlabel("Time (10-minute chunks of multiple days)")
	plt.ylabel("Bits")
	plt.title("Training data of "+str(total_weeks)+" total weeks (mon-fri)")# +str(end_week-begin_week+1)+" weeks")
	plt.show()

	normalized_tr = sp.std_normalization(bits_tr)
	plt.plot(normalized_tr)
	plt.xlabel("Time (10-minute chunks of multiple days)")
	plt.ylabel("Bits (normalized)")
	plt.title("Training data of "+str(total_weeks)+" total weeks (mon-fri)")# +str(end_week-begin_week+1)+" weeks")
	plt.show()

	bits_tst = dbp.week_data_prep(days_of_week[0], test_week, test_week, init_sample_rate)

	plt.plot(bits_tst)
	plt.xlabel("Time (minutes)")
	plt.ylabel("Bits")
	plt.title("Testing data")
	plt.show()

	normalized_tst = sp.std_normalization(bits_tst)
	plt.plot(normalized_tst)
	plt.xlabel("Time (minutes)")
	plt.ylabel("Bits")
	plt.title("Testing data")
	plt.show()
	#print("Test shape", bits_tst.shape)

	#Parameters for formatting training data
	window = 10
	validating = 0

	#Transforming the input data so that it can be used in a regressor
	Xtr, Ytr, Xtst, Ycomp, Xvalid, Yvalid = tr_data_prep(normalized_tr, normalized_tst, window, validating)

	print("x_training:", Xtr.shape, "\ty_training:", Ytr.shape)
	print("x_valid:", Xvalid.shape, "\ty_valid:", Yvalid.shape)
	print("x_test:", Xtst.shape, "\ty_tst:", Ycomp.shape)

	#Here begins the ridge regression

	'''
	for cvs in range(2, 15):
		best_ridge_regressor = RidgeCV(alphas=[1e-5, 1e-4, 1e-3, 1e-1, 1e0, 1e1, 1e3, 1e4], cv=cvs).fit(Xtr, Ytr)
		print("Best alpha for ", cvs, " folds: ", best_ridge_regressor.alpha_)
	'''
	best_ridge_regressor = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6], cv=12).fit(Xtr, Ytr)


	print("Ridge regresson best params: ", best_ridge_regressor.alpha_)
	print("Chi-squared test against training data: ", best_ridge_regressor.score(Xtr, Ytr))

	ridge_y_pred = best_ridge_regressor.predict(Xtst)

	ridge_mape_score = mape_test(Ycomp, ridge_y_pred) * 100
	print("MAPE score for ridge: ", ridge_mape_score)
	plot_ridge(ridge_y_pred, Ycomp, "Bits", day, str(window)+"\nand MAPE of "+str(ridge_mape_score))


	#Declaration of the Gaussian Process Regressor with its kernel parameters
	kernel = kernel_select("linear")
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                              	normalize_y=False, alpha=1e-3)

	print("Training the Gaussian Process...\n")
	gp.fit(Xtr, Ytr)
	print("Marginal likelihood:", gp.log_marginal_likelihood())
	print("Chi-squared test against training data: ", gp.score(Xtr,Ytr))
	y_self_pred, y_self_sigma = gp.predict(Xtr, return_std=True)
	print("self_pred: ", y_self_pred.shape, " ycomp: ", Ytr.shape, " self_sigma: ", y_self_sigma.shape)
	#print_gp(y_self_pred, y_self_sigma, np.array(Ytr), "Bits", day, str(window)+"\nMAPE: "+str(mape_test(Ytr, y_self_pred) * 100))
	#print(gp.get_params(deep=True))

	if(validating):
		print("Comparing with the validation set")
		y_valid_pred, y_valid_sigma = gp.predict(Xvalid, return_std=True)
		print("Chi-squared against validation data: ", gp.score(Xvalid, Yvalid))
		mape_valid_score = mape_test(Yvalid, y_valid_pred) * 100
		print("MAPE between actual and validation: ", mape_valid_score)
		plot_gp(y_valid_pred, y_valid_sigma, Yvalid, "Bits", day, str(window)+"\nMAPE: "+str(mape_valid_score))


	#Plotting prediction
	gp_y_pred, gp_y_sigma = gp.predict(Xtst, return_std=True)
	print("Chi-squared test against real data: ", gp.score(Xtst,Ycomp))
	mape_testing_score = mape_test(Ycomp, gp_y_pred) * 100
	print("MAPE between acutal and estimated:", mape_testing_score)
	plot_gp(gp_y_pred, gp_y_sigma, Ycomp, "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))

	plot_gp(gp_y_pred[:400], gp_y_sigma[:400], Ycomp[:400], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))
	plot_gp(gp_y_pred[600:700], gp_y_sigma[600:700], Ycomp[600:700], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))
	plot_gp(gp_y_pred[800:], gp_y_sigma[800:], Ycomp[800:], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))

	plt.plot(gp_y_sigma)
	plt.title("Standard deviation")
	plt.show()

	#one_sigma, two_sigma = verify_sigma(Ycomp, gp_y_pred, gp_y_sigma)
	one_sigma = verify_one_sigma(Ycomp, gp_y_pred, gp_y_sigma)
	two_sigma = verify_two_sigma(Ycomp, gp_y_pred, gp_y_sigma)
	if(one_sigma):
		print("Prediction is within 65% of the variance")
	else:
		print("Predicition is not with 65% of the variance")

	if(two_sigma):
		print("Prediction is within 95% of the variance")
	else:
		print("Predicition is not with 95% of the variance")

	#Coparing Ridge regression with the Gaussian process:
	ridge_mse = mean_squared_error(Ycomp, ridge_y_pred)
	gp_mse = mean_squared_error(Ycomp, gp_y_pred)
	print("Ridge mse: ", ridge_mse)
	print("GP mse: ", gp_mse)
