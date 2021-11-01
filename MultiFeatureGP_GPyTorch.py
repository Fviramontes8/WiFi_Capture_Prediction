# -*- coding: utf-8 -*-
""" 
Python 3.8.5 
Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch, gpytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on local files: 
    DatabaseConnector.py, 
    DatabaseProcessor.py,
    SignalProcessor.py,
    GPyTorchUtilities.py,
    PlotUtils.py
"""

# Private signal processor/sampler
import SignalProcessor as sp
# Private GPyTorch and PyTorch functions
import GPyTorchUtilities as gptu # Private plot functions
import PlotUtils as pu
# Private database processor
import DatabaseProcessor as dbp

# For matrix and linear algebra calcualtions
import numpy as np
#np.set_printoptions(threshold=np.nan)

import csv
# Mean squared error to determine quality of prediction
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

#from scipy.stats import ttest_ind

import torch
import gpytorch
import math

def buffer_filter(data, filter_window=3600):
	#Uniform window filtering and downsampling
	#filter_window=3600 # from 1 second to 1 hour (filter every 3600 seconds)
	X=sp.buffer2(data,filter_window,0).T
	Xs=np.sum(X,0)/filter_window
	return Xs

def save_as_csv(filename, data):
	with open(filename, "w") as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=",")
		for q in range(len(data)):
			csv_writer.writerow([data[q]])
			
def window_prep(data, window=10):
	# Bufferning to construct the data structure for the prediction
	#window=10
	X = sp.buffer2(data,window+1,window).T
	y = X[window,:].T
	X = X[:window,:]
	#x_list = list(X[0, :])
	#y_list = list(y)
	X = torch.Tensor(list(X))
	y = torch.Tensor(list(y))
	
	return X, y

if __name__ == "__main__":
	raw_nou_data = np.load("data/raw_nou_15weeks.npy")
	raw_bits_data = np.load("data/raw_bits_15weeks.npy")
	raw_pkt_data = np.load("data/raw_numpkt_15weeks.npy")
	raw_sig_data = np.load("data/raw_sigstrength_15weeks.npy")
	raw_phya_data = np.load("data/raw_phya_15weeks.npy")
	print(len(raw_nou_data))

	#pu.general_plot(raw_bits_data, "Raw bits")
	#pu.general_plot(raw_phya_data, "Raw phya")

	raw_data = [
		raw_nou_data,
		raw_bits_data, 
		raw_pkt_data, 
		raw_sig_data,
		raw_phya_data
	]
	
	raw_titles = [
		"Number of Users",
		"Bits",
		"Number of Packets",
		"Signal Strength",
		"802.11a data"
	]
	
	#pu.plot_features(raw_data, raw_titles)
	
	nou = buffer_filter(raw_nou_data)
	bits = buffer_filter(raw_bits_data)
	pktnum = buffer_filter(raw_pkt_data)
	sig = buffer_filter(raw_sig_data)
	phya = buffer_filter(raw_phya_data)


	nou = sp.std_normalization(nou)
	bits = sp.std_normalization(bits)
	pktnum = sp.std_normalization(pktnum)
	sig = sp.std_normalization(sig)
	phya = sp.std_normalization(phya)

	#pu.general_plot(bits, "All bits")
	#pu.general_plot(phya, "Phya bits")
	
	filtered_data = [nou, bits, pktnum, sig, phya]
	filter_str = "Filtered "
	filtered_titles = [
		filter_str+raw_titles[0],
		filter_str+raw_titles[1],
		filter_str+raw_titles[2],
		filter_str+raw_titles[3],
		filter_str+raw_titles[4]
	]
	training_features = raw_titles[:4]
		
	#pu.plot_features(filtered_data, filtered_titles)
	
	
	#pu.plot_autocorr(nou, "Auto correlation of number of users")
	#pu.plot_autocorr(bits, "Auto correlation of bits")
	#pu.plot_autocorr(pktnum, "Auto correlation of number of packets")
	#pu.plot_autocorr(sig, "Auto correlation of signal strength")
	#pu.plot_autocorr(phya, "Auto correlation of 802.11a bits")
	
	#pu.plot_crosscorr(bits, nou, "Cross correlation between number of users and bits")
	#pu.plot_crosscorr(bits, pktnum, "Cross correlation between bits and number of packets")
	#pu.plot_crosscorr(pktnum, sig, "Cross correlation between number of packets and signal strength")
	
	nou_tr_x, nou_tr_y = window_prep(nou)
	bits_tr_x, bits_tr_y = window_prep(bits)
	pktnum_tr_x, pktnum_tr_y = window_prep(pktnum)
	sig_tr_x, sig_tr_y = window_prep(sig)
	
	phya_tst_x, phya_tst_y = window_prep(phya)
	
	train_x = torch.stack([
			nou_tr_x.transpose(0, 1),
			bits_tr_x.transpose(0, 1),
			pktnum_tr_x.transpose(0, 1),
			sig_tr_x.transpose(0, 1)
		])
	print(train_x.shape)
	
	train_y = torch.stack([
			nou_tr_y,
			bits_tr_y,
			pktnum_tr_y,
			sig_tr_y
		])
	print(train_y.shape)
	
	#train_x = bits_tr_x.transpose(0, 1)
	#train_y = bits_tr_y
	
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	gp_model = gptu.LinearGPModel(train_x, train_y, likelihood)
	optimizer = torch.optim.Adam([
			{"params" : gp_model.parameters()},
		],
		lr = 0.5
	)
		
	gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 10)
	
	test_x = torch.stack([
		phya_tst_x.transpose(0, 1)
	])

	test_y = torch.stack([
		phya_tst_y
	])

	print(test_x.shape, test_y.shape)

	pred = gptu.TorchTest(test_x, gp_model, likelihood)
	#print(pred.mean.numpy().shape)
	#print("Confidence region shape:")
	#print(pred.confidence_region()[0].detach().numpy().shape)
	#print(pred.mean.numpy().shape[0])
		

	time_x = torch.Tensor([i for i in range(train_y.shape[1])])

    # This is for documentation
	pred_title = "Linear GP Predicting "
	x_title = "Time (hours)"
	y_title = "Bits"
	pu.PlotMTGPPred(
        time_x, 
        torch.Tensor(train_y), 
        time_x, 
        pred, 
        x_title, 
        y_title, 
        pred_title,
        training_features
    )
	for i in range(len(training_features)):
		gp_mape = sp.mape_test(train_y[i].numpy(), pred.mean[i].numpy())
		print(training_features[i], " mape: ", gp_mape)
		#print("Shape", train_y[i].numpy().shape, pred.mean[i].numpy().shape)

		gp_mse = mse(train_y[i].numpy(), pred.mean[i].numpy())
		print(training_features[i], " mse: ", gp_mse)

		gp_mae = mae(train_y[i].numpy(), pred.mean[i].numpy())
		print(training_features[i], " mae: ", gp_mae)

		gp_r_sq = r2_score(train_y[i].numpy(), pred.mean[i].numpy())
		print(training_features[i], " r2 score: ", gp_r_sq)
