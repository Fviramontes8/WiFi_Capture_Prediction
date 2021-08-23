# -*- coding: utf-8 -*-
"""
Python 3.8.5
Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch, gpytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
"""

# Private signal processor/sampler
import SignalProcessor as sp
# Private GPyTorch and PyTorch functions
import GPyTorchUtilities as gptu
# Private plot functions
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

from scipy.stats import ttest_ind

import torch
import gpytorch
import math

def buffer_filter(data, filter_window=1800):
	#Uniform window filtering and downsampling
	#filter_window=1800 # from 1 second to 1 hour (filter every 3600 seconds)
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

raw_nou_data = np.load("data/raw_nou_15weeks.npy")
raw_bits_data = np.load("data/raw_bits_15weeks.npy")
raw_pkt_data = np.load("data/raw_numpkt_15weeks.npy")
raw_sig_data = np.load("data/raw_sigstrength_15weeks.npy")
raw_phya_data = np.load("data/raw_phya_15weeks.npy")
print(len(raw_nou_data))

#pu.general_plot(raw_nou_data, "Number of users")
#pu.general_plot(raw_bits_data, "Bits")
#pu.general_plot(raw_pkt_data, "Number of packets")
#pu.general_plot(raw_sig_data, "Signal Strength")

#pu.general_plot(raw_phya_data, "802.11a data")

nou = buffer_filter(raw_nou_data)
bits = buffer_filter(raw_bits_data)
pktnum = buffer_filter(raw_pkt_data)
sig = buffer_filter(raw_sig_data)
phya = buffer_filter(raw_phya_data)

pu.general_plot(nou, "Filtered number of users")
pu.general_plot(bits, "Filtered bits")
pu.general_plot(pktnum, "Filtered number of packets")
pu.general_plot(sig, "Filtered signal strength")
pu.general_plot(phya, "Filtered 802.11a bits")

nou_tr_x, nou_tr_y = window_prep(nou)
bits_tr_x, bits_tr_y = window_prep(bits)
pktnum_tr_x, pktnum_tr_y = window_prep(pktnum)
sig_tr_x, sig_tr_y = window_prep(sig)
phya_tr_x, phya_tr_y = window_prep(phya)

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

likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_model = gptu.LinearGPModel(train_x, train_y, likelihood)
optimizer = torch.optim.Adam([
			{"params" : gp_model.parameters()},
		],
		lr = 0.1
	)
	
#gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 500)
#save_as_csv("raw_training_signalstrength_15weeks_nosample_singlecol.csv", x_list) 
#save_as_csv("raw_testing_signalstrength_15weeks_nosample_col.csv", y_list)