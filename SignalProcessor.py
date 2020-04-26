#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:31 2020
Packages needed: matplotlib, numpy, scipy

@author: frankie
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
"""

#For matrix and linear algebra calcualtions
import numpy as np

from math import sqrt

#For use of the butterworth and Savgol filters
from scipy import signal

#For ploting data
import matplotlib.pyplot as plt

def mean(values):
    '''Determines the mean of an array'''
    if len(values) > 0:
        return sum(values) / len(values)

def variance(values):
	'''Calculates and returns the variance of an array-like input'''
	mu = mean(values)
	value_sum = 0.0
	for i in values:
		value_sum += (i - mu) ** 2
	return value_sum / len(values)

def std_dev(values):
	'''Returns the standard deviation of an array-like input'''
	return sqrt(variance(values))

def std_normalization(values):
	'''Normalizes an array by subtacting the mean and dividing by the standard deviation'''
	mu = mean(values)
	sigma = std_dev(values)

	normalized_values = values[:]
	for i in range(len(normalized_values)):
		normalized_values[i] = (normalized_values[i] - mu) / sigma

	return normalized_values

def grab_nz(array, n ,z):
	'''Gets the first n to z values of an array, returns error if n is greater
		than the length of the array or if z < n or z > len(array)
	'''
	if (n <= len(array)) and (z <= len(array)):
		return np.atleast_1d([array[i] for i in range(n, z)]).T
	else:
		print("Usage: \n\tgrab_nz(array, n, z)\n\t\tn must be less than the length of array and n < z < len(array)")
		return None

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
	graphing = 0
	if graphing:
		plt.title("Subsampled data at a rate of " +str(sampling)+" for "+day)
		plt.ylabel(title)
		plt.xlabel("Time of day (seconds)")
		plt.plot(x_axis_xs, xs, "g", label="Sampled data")
		plt.legend()
		plt.show()
	return xs

def avg_sample(sample_arr, sample_size):
    '''Gets a chunk (like 5 values for example) takes the average of the chunk, then it is added to an array as one value. Continues to do this for the rest of the array.
    Example: array = [2, 2, 16, 4, 5, 15]
    result_value = avg_sample(array, 4)
    result_value is [6, 8]'''
    a = sample_arr
    sample_return = []
    j = 0
    k = sample_size - 1
    while k < len(a):
        m = []
        for i in range(j, k+1):
            m.append(a[i])
        sample_return.append(int(mean(m)))
        j += sample_size - 1
        k += sample_size - 1
        if k >= len(a):
            m = []
            k = len(a)
            for i in range(j+1, k):
                m.append(a[i])
            sample_return.append(int(mean(m)))
            break
    return np.atleast_1d(sample_return)

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