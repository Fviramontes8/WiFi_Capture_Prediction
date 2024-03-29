# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:35:15 2020

@author: Frankie
"""
import matplotlib.pyplot as plt
import numpy as np

# Private signal processor/sampler
import SignalProcessor as sp

# Graphs with PyTorch data
import torch
import GPyTorchUtilities as gptu

def PlotMTGPPred(XCompare, YCompare, XPred, YPred, Xtitle="", Ytitle="", Title="", TasksTitle=""):
	with torch.no_grad():
		#fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		mt_lower_sigma, mt_upper_sigma = YPred.confidence_region()
		for i in range(YPred.mean.numpy().shape[0]):
			sigma1_lower, sigma1_upper = gptu.ToStdDev1MT(
				YPred.mean.numpy()[0],
				mt_lower_sigma[i],
				mt_upper_sigma[i]
			)

			plt.plot(XCompare.numpy(), YCompare.numpy()[i], "k")
			plt.plot(XPred.numpy(), YPred.mean.numpy()[i], "b")
			#print(XPred.numpy().shape)
			#print(mt_lower_sigma.numpy().shape)
			#print(mt_upper_sigma.numpy().shape)
			#with 1 standard deviation\nand 2 standard deviations"
			plt.fill_between(
				XPred.numpy()[i],
				mt_lower_sigma.numpy()[i],
				mt_upper_sigma.numpy()[i],
				alpha = 0.5
			)
	
			plt.fill_between(
				XPred.numpy(),
				sigma1_lower,
				sigma1_upper,
				alpha = 0.5
			)
	
			plt.xlabel(Xtitle)
			plt.ylabel(Ytitle)
			plt.title(Title+TasksTitle[i])
			plt.legend(["Observed Data", "Prediction Mean", "2 StdDev Confidence", "1 StdDev Confidence"])
			plt.show()

def PlotGPPred(XCompare, YCompare, XPred, YPred, Xtitle="", Ytitle="", Title=""):
	with torch.no_grad():
		fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		lower_sigma, upper_sigma = YPred.confidence_region()

		sigma1_lower, sigma1_upper = gptu.ToStdDev1(YPred)

		ax.plot(XCompare.numpy(), YCompare.numpy(), "k")
		ax.plot(XPred.numpy(), YPred.mean.numpy(), "b")
		ax.fill_between(
			XPred.numpy(),
			lower_sigma.numpy(),
			upper_sigma.numpy(),
			alpha = 0.5
		)

		ax.fill_between(
			XPred.numpy(),
			sigma1_lower,
			sigma1_upper,
			alpha = 0.5
		)

		ax.set_xlabel(Xtitle)
		ax.set_ylabel(Ytitle)
		ax.set_title(Title)
		ax.legend(["Observed Data", "Prediction Mean", "2 StdDev Confidence", "1 StdDev Confidence"])
		plt.show()
		
def general_plot(data, title="", xtitle="", ytitle=""):
	plt.plot(data)
	plt.title(title)
	plt.xlabel(xtitle)
	plt.ylabel(ytitle)
	plt.show()
	
def general_double_plot(data1, data2, title="", xtitle="", ytitle=""):
	assert(len(data1) == len(data2))
	plt.plot(data1)
	plt.plot(data2)
	plt.title(title)
	plt.xlabel(xtitle)
	plt.ylabel(ytitle)
	plt.show()

def plot_autocorr(data, titles):
	for i in range(len(data)):
		self_corr = np.correlate(data[i], data[i], "full")
		general_plot(self_corr, titles[i])

def plot_crosscorr(x, y, title):
	assert(len(x)==len(y))
	crosscorr = np.correlate(x, y, "full")
	general_plot(crosscorr, title)

def plot_features(feats, feat_titles):
	assert(len(feats)==len(feat_titles))
	for i in range(len(feats)):
		general_plot(feats[i], feat_titles[i])
		
def plot_gp(pred, sigma, compare, x_title, y_title, day, window):
	#print("Arguement size: ", pred.shape, sigma.shape, compare.shape)
	#print("Feature: ", feature, "\nDay: ", day, "\nTitle ", window)
	stddev_coef = 0.98#1.96
	var_coef = 1.96
	prediction_time= [p+1 for p in range(len(pred))]
	plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
			  np.concatenate([pred-var_coef*sigma,
					 (pred+var_coef*sigma)[::-1]]),
			  alpha=.5, fc='b', ec='none', label="2 StdDev confidence")
	plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
			  np.concatenate([pred-stddev_coef*sigma,
					 (pred+stddev_coef*sigma)[::-1]]),
			  alpha=.5, fc='m', ec='none', label="1 StdDev confidence")
	plt.plot(prediction_time, pred, "c-", label="GP Prediction")
	plt.plot(prediction_time, compare, "y-", label="Actual data")
	plt.legend()
	plt.title("Gaussian Process Prediction with 6th order Butterworth filtering,\nPredicting "
		   +day+"\nWith window of "+str(window)+"\nAnd 1 and 2 standard deviations")
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.show()
	
def plot_ridge(pred, compare, feature, day, window):
	prediction_time= [p+1 for p in range(len(pred))]

	plt.plot(prediction_time, pred, "c-", label="Ridge Regression Prediction")
	plt.plot(prediction_time, compare, "y-", label="Actual data")
	plt.legend()
	plt.title("Ridge Regression Prediction with 6th order Butterworth filtering,\nPredicting a "
		   +day+"\nWith window of "+str(window))
	plt.xlabel("Time (hours)")
	plt.ylabel(feature+" (predicted)")
	plt.show()
	
def plot_ridge_prediction(pred, ycomp, day, window):
	ridge_mape_score = sp.mape_test(ycomp, pred)
	#print("MAPE score for ridge: ", ridge_mape_score)
	plot_ridge(pred, ycomp, "Bits", day, str(window)+"\nand MAPE of "+str(ridge_mape_score))
