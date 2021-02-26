# -*- coding: utf-8 -*-
"""
@author: Frankie
"""
#Machine Learning package for the Gaussian Process Regressor
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ, RBF

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