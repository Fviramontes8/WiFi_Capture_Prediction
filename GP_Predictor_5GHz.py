#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.6
Packages needed: scikit-learn, psycopg2, numpy, scipy
@author: Francisco Viramontes
"""
#Package to interface with AWS database
import DatabaseConnector as dc

#For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

#For ploting data
import matplotlib.pyplot as plt

#For use of the butterworth and Savgol filters
from scipy import signal

#Machine Learning package for the Gaussian Process Regressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ, RBF
#import commpy.filters as comm

def grab_nz(array, n ,z):
	'''Gets the first n to z values of an array, returns error if n is greater
		than the length of the array or if z < n or z > len(array)
	'''
	if (n <= len(array)) and (z <= len(array)):
		return np.atleast_1d([array[i] for i in range(n, z)]).T
	else:
		print("Usage: \n\tgrab_nz(array, n, z)\n\t\tn must be less than the length of array and n < z < len(array)")
		return -1

def GP_Prep(training, testing, window, validating=0):
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
		x_valid = np.atleast_2d([grab_nz(validation, m, n) for m, n in zip(range(validation.shape[0]), range(window, validation.shape[0]))])
		y_valid = np.atleast_2d([[validation[i] for i in range(window, validation.shape[0])]]).T

	Xtr = np.atleast_2d([grab_nz(training, m, n) for m, n in zip(range(training.shape[0]), range(window, training.shape[0]))])
	Ytr = np.atleast_2d([[training[i] for i in range(window, training.shape[0])]]).T
	Xtst = np.atleast_2d([grab_nz(testing, m, n) for m, n in zip(range(testing.shape[0]), range(window, testing.shape[0]))])
	Ycomp = np.atleast_2d([testing[i] for i in range(window, testing.shape[0])]).T

	return Xtr, Ytr, Xtst, Ycomp, x_valid, y_valid

def print_gp(pred, sigma, compare, feature, day, window):
	#print("Arguement size: ", pred.shape, sigma.shape, compare.shape)
	#print("Feature: ", feature, "\nDay: ", day, "\nTitle ", window)
	sigma_coef = 0.98#1.96
	prediction_time= [p+1 for p in range(len(pred))]
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(prediction_time, compare, "y-", label="Validation data")
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

def read_5ghz_day(table_name):
	'''
	Input: A string that describes a table name.
	Output: A touple of lists.
	Description: Gets the contents of the table if it exists. For this function
		specifically, there is an assumption that the table has a format of
		such: it is a nx9 table with column names (Key, ts, nou, bits, pkt_num,
		sigs, dr, phya, phyn) and is in the file 'databse.ini'. Please look at
		documentation for DatabaseConnect() and _config() for more information.
	'''
	db = dc.DatabaseConnect()
	db.connect()
	data = db.readTable(table_name)
	#print("Next key is:", db.getNextKey(table_name))
	db.disconnect()

	t_stamps = []
	num_of_users = []
	bits = []

	for db_iter in sorted(data, key=lambda dummy_arr:dummy_arr[1]):
		t_stamps.append(db_iter[1])
		num_of_users.append(db_iter[2])
		bits.append(db_iter[3])

	return_data = [t_stamps,
					num_of_users,
					bits
					]
	return return_data

def butterfilter(input_arr, title, day, freq=60):
	'''
	Input:
		A list that can be represented as a time series that is the feature
			desired to be filtered (input_arr)
	Output:
		A new list that are a filtered version of the input. It is the
			same length as the input
	'''
	z = (0.9/4) / freq
	begin_cutoff = 0
	b, a = signal.butter(6, z, 'low')
	xf = signal.lfilter(b, a, input_arr)
	graphing = 0
	if graphing:
		plt.plot(input_arr[begin_cutoff:], label="Original Data")
		plt.plot(xf[begin_cutoff:], label="Filtered Data")
		plt.title("Filtered "+title+" for "+day)
		plt.ylabel(title)
		plt.xlabel("Time of day (seconds)")
		plt.legend()
		plt.show()
	return xf

def sub_sample(xf, title, day, sampling=60):
	'''
	Input:
		A string that describes the filter (title)
		A sampling frequency (sampling) [the default value is 60 to sample
			the data into minute chunks]
	Output: A list of filtered data points (1/sampling) of original size
	'''
	xf_copy = np.array(xf).copy()
	xs = xf_copy[1::sampling]
	x_axis_xs = np.array([i for i in range(1, len(xf), sampling)])
	graphing = 1
	if graphing:
		plt.title("Subsampled data at a rate of " +str(sampling)+" for "+day)
		plt.ylabel(title)
		plt.xlabel("Time of day (seconds)")
		plt.plot(x_axis_xs, xs, "g", label="Sampled data")
		plt.legend()
		plt.show()
	return xs

def savgol(input_arr, title):
	'''
	DEPRECATED
	'''
	sampling = 60
	xf = signal.savgol_filter(input_arr, 5, 2)
	xf_copy = np.array(xf).copy()
	xs = xf_copy[1::sampling]
	x_axis_xs = np.array([i for i in range(len(xf))])
	x_axis_xs = x_axis_xs[::sampling]
	plt.plot(input_arr, label="Original Data")
	plt.plot(x_axis_xs, xs, label="Filtered Data")
	plt.title("Time series and filtered data for "+title)
	plt.ylabel(title)
	plt.xlabel("Time of day (seconds)")
	plt.legend()
	plt.show()
	return xs

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
	result /= size
	return result

def kernel_select(kernel_str):
	if type(kernel_str) is not str:
		print("Input is not a string!\n")
		return None
	if kernel_str == "linear":
		kernel1 = LK(sigma_0 = 10, sigma_0_bounds=(10e-1, 10e1))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(noise_level=1e-2, noise_level_bounds = (10e-4, 10e-2))
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
	elif kernel_str == "RBF":
		kernel1 = RBF(length_scale=5, length_scale_bounds=(1e-1, 1e1))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(0.1)
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
	else:
		kernel1 = LK(sigma_0 = 1, sigma_0_bounds=(10e-1, 10e1))
		kernel2 = CK(constant_value=1)
		kernel3 = WK(noise_level=0.01, noise_level_bounds = (10e1, 10e3))
		kernel = Sum(kernel1, kernel2)
		kernel = Sum(kernel, kernel3)
		print("Not a valid kernel name, defaulting to ", kernel)

	#print(kernel)
	return kernel

def week_data_prep(day, start_week, end_week, sample_rate, sample_rate2=None):
	training_data = []
	labels_5ghz = ["Number of users",
       				"Bits"
       				]

	for week in range(start_week, end_week+1):
		table_name = "5pi_"+str(day)+str(week)
		day_data = read_5ghz_day(table_name)

		while(day_data[2][0] < 1):
				del day_data[2][0]

		filtered_data = butterfilter(day_data[2], labels_5ghz[1], table_name)
		sampled_data = sub_sample(filtered_data, labels_5ghz[1], table_name, sample_rate)
		if(sample_rate2):
			sampled_data = sub_sample(sampled_data, labels_5ghz[1], table_name, sample_rate2)
		training_data.extend(sampled_data)

	training_data = np.array(training_data)
	return training_data

def day_data_prep(days_of_week, num_of_weeks, sample_rate, sample_rate2=None):
	training_data = []
	labels_5ghz = ["Number of users",
       				"Bits"
       				]
	for week_num in range(2, num_of_weeks+1):
		for day in days_of_week:
			table_name = "5pi_"+str(day)+str(week_num)
			day_data = read_5ghz_day(table_name)

			while(day_data[2][0] < 1):
				del day_data[2][0]
			filtered_data = butterfilter(day_data[2], labels_5ghz[1], table_name)
			sampled_data = sub_sample(filtered_data, labels_5ghz[1], table_name, sample_rate)
			if(sample_rate2):
				sampled_data = sub_sample(sampled_data, labels_5ghz[1], table_name, sample_rate2)
			training_data.extend(sampled_data)

	training_data = np.array(training_data)
	return training_data

def verify_sigma(pred_series, sigma_series):
	upper_sigma = []
	for i in range(len(pred_series)):
		upper_sigma.append(pred_series[i] + sigma_series[i])

	lower_sigma = []
	for j in range(len(pred_series)):
		lower_sigma.append(pred_series[j] - sigma_series[j])

	count = 0
	for h in range(len(pred_series)):
		if((pred_series[h] < upper_sigma[h]) & (pred_series[h] > lower_sigma[h])):
			count += 1

	one_sigma=False
	two_sigma=False
	if( count > (len(pred_series) * 0.65)):
		one_sigma = True

	if( count > (len(pred_series) * 0.95)):
		two_sigma = True

	return one_sigma, two_sigma

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
	second_sample_rate = 6
	test_week=15

	#bits_tr = week_data_prep(day, begin_week, end_week, init_sample_rate, second_sample_rate)
	bits_tr = day_data_prep(days_of_week, 6, init_sample_rate, second_sample_rate)
	plt.plot(bits_tr)
	plt.xlabel("Time (10-minute chunks of multiple days)")
	plt.ylabel("Bits")
	plt.title("Training data of "+str(10)+" total weeks (mon-fri)")# +str(end_week-begin_week+1)+" weeks")
	plt.show()

	bits_tst = week_data_prep(days_of_week[3], test_week, test_week, init_sample_rate)

	plt.plot(bits_tst)
	plt.xlabel("Time (minutes)")
	plt.ylabel("Bits")
	plt.title("Testing data")
	plt.show()
	#print("Test shape", bits_tst.shape)

	#Declaration of the Gaussian Process Regressor with its kernel parameters
	kernel = kernel_select("linear")
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                              	normalize_y=False)#, alpha=1e-3)
	window = 10
	validating = 0

	#Transforming the input data so that it can be used in the Gaussian Process
	Xtr, Ytr, Xtst, Ycomp, Xvalid, Yvalid = GP_Prep(bits_tr, bits_tst, window, validating)

	print("x_training:", Xtr.shape, "\ty_training:", Ytr.shape)
	print("x_valid:", Xvalid.shape, "\ty_valid:", Yvalid.shape)
	print("x_test:", Xtst.shape, "\ty_tst:", Ycomp.shape)


	print("Training the Gaussian Process...\n")
	gp.fit(Xtr, Ytr)
	print("Marginal likelihood:", gp.log_marginal_likelihood())
	print("Chi-squared test against training data: ", gp.score(Xtr,Ytr))
	y_self_pred, y_self_sigma = gp.predict(Xtr, return_std=True)
	print("self_pred: ", y_self_pred.shape, " ycomp: ", Ytr.shape, " self_sigma: ", y_self_sigma.shape)
	#print_gp(y_self_pred, y_self_sigma, np.array(Ytr), "Bits", day, str(window)+"\nMAPE: "+str(mape_test(Ytr, y_self_pred) * 100))
	print(gp.get_params(deep=True))
	if(validating):
		print("Comparing with the validation set")
		y_valid_pred, y_valid_sigma = gp.predict(Xvalid, return_std=True)
		print("Chi-squared against validation data: ", gp.score(Xvalid, Yvalid))
		mape_valid_score = mape_test(Yvalid, y_valid_pred) * 100
		print("MAPE between actual and validation: ", mape_valid_score)
		print_gp(y_valid_pred, y_valid_sigma, Yvalid, "Bits", day, str(window)+"\nMAPE: "+str(mape_valid_score))


	#Plotting prediction
	y_pred, y_sigma = gp.predict(Xtst, return_std=True)
	print("Chi-squared test against real data: ", gp.score(Xtst,Ycomp))
	mape_testing_score = mape_test(Ycomp, y_pred) * 100
	print("MAPE between acutal and estimated:", mape_testing_score)
	print_gp(y_pred, y_sigma, Ycomp, "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))

	#print_gp(y_pred[:400], y_sigma[:400], Ycomp[:400], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))

	#print_gp(y_pred[400:800], y_sigma[400:800], Ycomp[400:800], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))
	#print_gp(y_pred[800:], y_sigma[800:], Ycomp[800:], "Bits", day, str(window)+"\nMAPE: "+str(mape_testing_score))

	one_sigma, two_sigma = verify_sigma(y_pred, y_sigma)
	if(one_sigma):
		print("Prediction is within 65% of the variance")
	else:
		print("Predicition is not with 65% of the variance")

	if(two_sigma):
		print("Prediction is within 95% of the variance")
	else:
		print("Predicition is not with 95% of the variance")
