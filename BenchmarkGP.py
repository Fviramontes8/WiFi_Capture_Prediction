# -*- coding: utf-8 -*-
"""
This program benmarks a simple Gaussian Process
@author: Frankie
"""

#For matrix and linear algebra calcualtions
import numpy as np

#For ploting data
import matplotlib.pyplot as plt

#Scikit learn packages for ML tools
from sklearn.metrics import mean_squared_error

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum, RBF

def plot_gp(pred, sigma, compare, plot_title):
	# Coefficient for calculation 1 standard deviation
	sigma_coef = 0.98
	
	# X-axis for predition data
	prediction_time= np.linspace(-10, 10, len(pred))
	
	# X-axis for actual data
	compare_time = np.linspace(-2, 2, len(compare))
	
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(compare_time, compare, "k.", label="Actual data")
	plt.plot(prediction_time, pred+sigma_coef*sigma, "k--", label="Standard Deviation")
	plt.plot(prediction_time, pred-sigma_coef*sigma, "k--")
	plt.legend()
	plt.title("Gaussian Process Prediction\n"+str(plot_title)+"\nAnd 1 standard deviation")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim(-6, 6)
	plt.ylim(-10, 10)
	plt.show()

	sigma_coef = 1.96
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(compare_time, compare, "k.", label="Validation data")
	plt.plot(prediction_time, pred+sigma_coef*sigma, "k--", label="Standard Deviation")
	plt.plot(prediction_time, pred-sigma_coef*sigma, "k--")
	plt.legend()
	plt.title("Gaussian Process Prediction\n"+str(plot_title)+"\nAnd two standard deviations")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim(-6, 6)
	plt.ylim(-10, 10)
	plt.show()

def mape_test(actual, estimated):
	if( (type(actual) is np.ndarray) & (type(estimated) is np.ndarray)):
		pass
	else:
		print("Inputs must be data type: numpy.ndarray")
		return None;
	"""
	size = len(actual)
	result = 0.0
	for i in range(size):
		result += np.abs( (actual[i] - estimated[i])  / actual[i] )
	result /= size"""
	result = np.mean(np.abs((actual-estimated)/actual))
	return float(result)

def kernel_select(kernel_str):
	if type(kernel_str) is not str:
		print("Input is not a string! Defaulting to a linear kernel\n")
	if kernel_str == "linear":
		kernel1 = LK(sigma_0 = 1, sigma_0_bounds=(10e-3, 10e3))
		kernel2 = CK(constant_value=1e-4)
		kernel3 = WK(noise_level=10e0, noise_level_bounds = (10e-5, 10e-1))
		kernel = Sum(kernel1, kernel3)
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

	return sigma_one_count/len(actual_series)

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

	return sigma_two_count/len(actual_series)

if __name__ == "__main__":
	Xtr = np.linspace(-2, 2, 10)
	Ytr = np.random.normal(0, 1, 10)
	
	plt.title("Inital data")
	plt.plot(Ytr, Xtr, ".")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim(-6, 6)
	plt.ylim(-10, 10)
	plt.show()
	
	Xtr = Xtr.reshape(1, -1).T
	Xtst = np.linspace(-10, 10, 100).reshape(1, -1).T
	
	kernel = kernel_select("linear")
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                              	normalize_y=False, alpha=1e-3)
	
	print("Training the Gaussian Process...\n")
	gp.fit(Xtr, Ytr)
	print("Marginal likelihood:", gp.log_marginal_likelihood())
	y_self_pred, y_self_sigma = gp.predict(Xtst, return_std=True)
	print("self_pred: ", y_self_pred.shape, " ycomp: ", Ytr.shape, " self_sigma: ", y_self_sigma.shape)
	
	# Slicing overlapping data with training data
	Y_lower_slice = int((len(y_self_pred)/2) - (len(Ytr)/2))
	Y_upper_slice = int((len(y_self_pred)/2) + (len(Ytr)/2))
	
	plot_gp(y_self_pred, y_self_sigma, np.array(Ytr), "MAPE: "+str(mape_test(Ytr, y_self_pred[Y_lower_slice:Y_upper_slice]) * 100))

	#Sigma verification for the first prediction
	Y_lower_slice = int((len(y_self_pred)/2) - (len(Ytr)/2))
	Y_upper_slice = int((len(y_self_pred)/2) + (len(Ytr)/2))
	one_sigma = verify_one_sigma(Ytr, y_self_pred[Y_lower_slice:Y_upper_slice], y_self_sigma[Y_lower_slice:Y_upper_slice])
	two_sigma = verify_two_sigma(Ytr, y_self_pred[Y_lower_slice:Y_upper_slice], y_self_sigma[Y_lower_slice:Y_upper_slice])
	print(one_sigma,"is contained within 1 standard deviation")
	print(two_sigma, "is contained within 2 standard deviations")
	