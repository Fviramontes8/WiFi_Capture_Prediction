# -*- coding: utf-8 -*-
""" Python 3.8.5 Packages needed: scikit-learn, psycopg2, numpy, scipy, pytorch, gpytorch
@author: Francisco Viramontes
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on local files: DatabaseConnector.py, DatabaseProcessor.py Signal Processor.py
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

def plot_autocorr(data, title):
	self_corr = np.correlate(data, data, "full")
	pu.general_plot(self_corr, title)

def plot_crosscorr(x, y, title):
	crosscorr = np.correlate(x, y, "full")
	pu.general_plot(crosscorr, title)

def plot_features(feats, feat_titles):
	assert(len(feats)==len(feat_titles))
	for i in range(len(feats)):
		pu.general_plot(feats[i], feat_titles[i])

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

plot_features(raw_data, raw_titles)

nou = buffer_filter(raw_nou_data)
bits = buffer_filter(raw_bits_data)
pktnum = buffer_filter(raw_pkt_data)
sig = buffer_filter(raw_sig_data)
phya = buffer_filter(raw_phya_data)

filtered_data = [nou, bits, pktnum, sig, phya]
filter_str = "Filtered "
filtered_titles = [
	filter_str+raw_titles[0],
	filter_str+raw_titles[1],
	filter_str+raw_titles[2],
	filter_str+raw_titles[3],
	filter_str+raw_titles[4]
]
	
plot_features(filtered_data, filtered_titles)

'''
pu.general_plot(nou, "Filtered number of users")
pu.general_plot(bits, "Filtered bits")
pu.general_plot(pktnum, "Filtered number of packets")
pu.general_plot(sig, "Filtered signal strength")
pu.general_plot(phya, "Filtered 802.11a bits")
'''

#plot_autocorr(nou, "Auto correlation of number of users")
#plot_autocorr(bits, "Auto correlation of bits")
#plot_autocorr(pktnum, "Auto correlation of number of packets")
#plot_autocorr(sig, "Auto correlation of signal strength")
#plot_autocorr(phya, "Auto correlation of 802.11a bits")

#plot_crosscorr(bits, nou, "Cross correlation between number of users and bits")
#plot_crosscorr(bits, pktnum, "Cross correlation between bits and number of packets")
#plot_crosscorr(pktnum, sig, "Cross correlation between number of packets and signal strength")

nou_tr_x, nou_tr_y = window_prep(nou)
bits_tr_x, bits_tr_y = window_prep(bits)
pktnum_tr_x, pktnum_tr_y = window_prep(pktnum)
sig_tr_x, sig_tr_y = window_prep(sig)

phya_tst_x, phya_tst_y = window_prep(phya)

"""
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
"""
	
#gptu.TorchTrain(train_x, train_y, gp_model, likelihood, optimizer, 500)
#save_as_csv("raw_training_signalstrength_15weeks_nosample_singlecol.csv", x_list) 
#save_as_csv("raw_testing_signalstrength_15weeks_nosample_col.csv", y_list)
