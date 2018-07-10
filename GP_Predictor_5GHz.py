#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Created on Fri Sep 01 11:06:34 2017

@author Francisco Viramontes
'''
import DatabaseConnector as dc
import numpy as np
np.set_printoptions(threshold=np.nan)
#import scipy as sp
#import pylab as pb
import matplotlib.pyplot as plt
#from random import seed
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum
#from sklearn.gaussian_process.kernels import RationalQuadratic as RQ, ExpSineSquared as ESS 

def mean(values):
    '''Determines the mean of an array'''
    if len(values) > 0:
        return sum(values) / float(len(values))
    else:
        print("Usage:\n\tmean(values)\n\t\tvalues is an array of length greater than 0")
        return 0

def sample_var(values, sample_mean=None):
    '''Takes the sum of means and divides by the length of the array'''
    if sample_mean == None:
        sample_mean = mean(values)
    return sum([(x-sample_mean)**2 for x in values])/len(values)

def covariance(x, mean_x, y, mean_y):
    '''Determines the covariance of two functions, returns 0 otherwise'''
    cov = 0.0
    for i in range(len(x)):
        cov += (x[i] - mean_x) * (y[i] - mean_y)
    return cov

def sub_sample(sample_arr, sample_size):
    '''Gets every nth elements, it also includes the last element of the array'''
    new_sample = sample_arr
    return_sample = []
    q = (sample_size - 1)
    while q < len(sample_arr):
        return_sample.append(new_sample[q])
        q += sample_size
        
        if q >= len(sample_arr) - 1:
            '''
            q = len(sample_arr) - 1
            return_sample.append(new_sample[q])
            '''
            return_sample.append(new_sample[-1])
            break
    return return_sample

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

def grab_n(array, n):
    '''Gets the first n values of an array, returns error if n is greater than the length of the array'''
    if n < len(array):
        return np.atleast_1d([array[i] for i in range(n)]).T
    else:
        print("Usage: \n\tgrab_n(array, n), n must be less than the length of array")
        return 0

def grab_nz(array, n ,z):
    '''Gets the first n values of an array, returns error if n is greater than the length of the array or if z < n or z > len(array)'''
    if n <= len(array):
        if z <= len(array):
            return np.atleast_1d([array[i] for i in range(n, z)]).T
        else:
            print("Usage: \n\tgrab_nz(array, n, z)\n\t\tn must be less than the length of array and n < z < len(array)")
            return 0
    else:
        print("Usage: \n\tgrab_nz(array, n, z)\n\t\tn must be less than the length of array")
        return 0

#(nou, nou_tst, 60, 0, 100, 15)
def GP_prep(train, test, avg_samp, sub_samp_begin, sub_samp_end, window):
    '''Inputs: train/test, which needs to be an array and each can be a different size, avg_samp is 
     an integer that is going to sub-sample train/test (ex. if avg_samp = 60, it will sample every 60 values as one) so 
     avg_samp < length of train/test, sub_samp_begin/sub_samp_end specifies the total bounds of what the user wants 
     to keep from the sub-sampled result from avg_samp, so sub_samp_begin/sub_samp_end < length of train/test, the window
     specifies how wide the resultant matrix of this function is.
     Output: Length of total sampled values, training and test matricies that has window of averaged sampled values,
         and a graph-able array of what the training values look like (ycomp)
     Example: window = 5, length of array input (both train and test are same size in this example) = n
         [x_0 x_1 ... x_4]          [x_5]
         [x_1 x_2 ... x_5]          [x_6]
     x = [x_2 x_3 ... x_6]     y =  [x_7]
         [... ... ... ...]          [...]
         [x_n-5-1... ... x_n-1]     [x_n]
         '''
    samp_train = avg_sample(train, avg_samp)
    tot_samp = len(samp_train)
    tr = np.atleast_2d([grab_nz(samp_train, m, n) for m, n in zip(range(samp_train.shape[0]), range(window, samp_train.shape[0]))])
    Xtr = np.atleast_2d(tr)
    ob = np.atleast_2d([[samp_train[i] for i in range(window, samp_train.shape[0])]])
    Ytr = np.atleast_2d(ob).T 
    samp_test = avg_sample(test, avg_samp) 
    samp_test1 = grab_nz(samp_test, sub_samp_begin, sub_samp_end) 
    feat_xtest = [grab_nz(samp_test1, m, n) for m, n in zip(range(samp_test1.shape[0]), range(window, samp_test1.shape[0]))]
    xtst =  np.atleast_2d(feat_xtest)
    feat_comp = [samp_train[i] for i in range(window, samp_test1.shape[0])]
    ycomp = np.atleast_2d(feat_comp).T
    feat_ytest = [samp_test[i] for i in range(window, samp_test1.shape[0])]
    ytst = np.atleast_2d(feat_ytest).T
    return tot_samp, Xtr, Ytr, xtst, ycomp, ytst, samp_train

def feature_plot(yaxis, feature, color):
    #Take a look to https://matplotlib.org/2.0.2/api/pyplot_api.html#matplotlib.pyplot.plot for color options.
    plt.title("Sampled feature: " + str(yaxis) + " with half-hour sample rate")
    plt.xlabel("Time in half-hours")
    plt.ylabel(yaxis)
    plt.plot(feature, color + "--")
    plt.legend()
    plt.show()
    
#Main:
timestamps = []
timestamps_tst = []
nou = []
nou_tst = []
bits = []
bits_tst = []
pktNum = []
pkt_tst = []
sigS = []
sigS_tst = []
dataRate = []
dR_tst = []
phyA = []
a_tst = []
phyN = []
n_tst = []

db = dc.DatabaseConnect()
#db.connect()
#print("Hello")
db.test_connect(database_name="postgres", username="postgres", host_name="129.24.26.110", password_name="Cerculsihr4T")
#print("Henlo")
train_table = "5pi_sun"
test_table = "5pi_sun2"

train = db.readTable(train_table)
test = db.readTable(test_table)

#create_table_name = "first_test3"
#db.createDataTable_5ghz(create_table_name)
#print(db.getTableNames())

#table_name_test = "5pi_thurs2"
#table_name_test2 = "5pi_fri2"
#test = db.readTable(table_name_test)
#db.createDataTable_5ghz(table_name_test)
#db.createDataTable_5ghz(table_name_test2)

db.disconnect()

#Data from table (in form of tuple)
#k and l are just dummy arrays
for k in sorted(train, key=lambda hello: hello[1]):
    timestamps.append(int(k[1]))
    nou.append(int(k[2]))
    bits.append(int(k[3]))
    pktNum.append(int(k[4]))
    sigS.append(int(k[5]))
    dataRate.append(int(k[6]))
    phyA.append(int(k[7]))
    phyN.append(int(k[8]))

    
for l in sorted(test, key=lambda yello: yello[1]):
    timestamps_tst.append(int(l[1]))
    nou_tst.append(int(l[2]))
    bits_tst.append(int(l[3]))
    pkt_tst.append(int(l[4]))
    sigS_tst.append(int(l[5]))
    dR_tst.append(int(l[6]))
    a_tst.append(int(l[7]))
    n_tst.append(int(l[8]))


#print "Average number of users: " + str(int(mean(nou)))
#print "Standard deviation: " + str(int(np.sqrt(sample_var(nou, mean(nou)))))

training_data=[nou, 
               bits, 
               pktNum, 
               sigS, 
               dataRate, 
               phyA,
               phyN
               ]

test_data = [nou_tst, 
             bits_tst, 
             pkt_tst, 
             sigS_tst, 
             dR_tst, 
             a_tst, 
             n_tst
             ]

labels = ["Number of users", 
          "Bits", 
          "Number of Packets", 
          "Signal Strength",
          "Data Rate(MB)", 
          "802.11a bits", 
          "802.11n bits"
          ]

labels_5ghz = ["Number of users",
               "Bits",
               "Number of Packets",
               "Signal Strength",
               "Data Rate(MB)",
               "802.11a bits",
               "802.11n bits"
               ]



print(len(nou))
for iter_1, iter_label in zip(training_data, labels):
    plt.plot(timestamps, iter_1, "r-")
    plt.ylabel("Feature: "+iter_label)#("Number of users")
    plt.xlabel("Timestamp")
    plt.show()

print(len(nou_tst))
for iter_2, iter_label2 in zip(test_data, labels_5ghz):
    plt.plot(timestamps_tst, iter_2, "c-")
    plt.ylabel("Feature: "+iter_label2)#("Number of users")
    plt.xlabel("Timestamp")
    plt.show()


'''
test_upload = []
    
for p in range(len(timestamps)):
    db.test_connect(database_name="postgres", username="postgres", host_name="129.24.26.110", password_name="Cerculsihr4T")
    key_holder = db.getNextKey(table_name_test)
    print(key_holder)
    if(key_holder == None):
        key_holder = 0
    test_upload = [str(p), str(timestamps[p]), str(nou[p]), str(bits[p]), str(pktNum[p]), str(sigS[p]), str(dataRate[p]), str(phyB[p]), str(phyN[p])]
    print(test_upload)
    db.writeData_5ghz(table_name_test, test_upload)
    db.disconnect()

test_upload2 = []
    
for p in range(len(timestamps_tst)):
    db.test_connect(database_name="postgres", username="postgres", host_name="129.24.26.110", password_name="Cerculsihr4T")
    key_holder = db.getNextKey(table_name_test2)
    print(key_holder)
    if(key_holder == None):
        key_holder = 0
    test_upload2 = [str(p), str(timestamps_tst[p]), str(nou_tst[p]), str(bits_tst[p]), str(pkt_tst[p]), str(sigS_tst[p]), str(dR_tst[p]), str(b_tst[p]), str(n_tst[p])]
    print(test_upload2)
    db.writeData_5ghz(table_name_test2, test_upload2)
    db.disconnect()
'''
'''#Uncomment to see sampled features
sample_rate = 1800

sampled_nou = avg_sample(nou, sample_rate)
sampled_bits = avg_sample(bits, sample_rate)
sampled_pktNum = avg_sample(pktNum, sample_rate)
sampled_sigS = avg_sample(sigS, sample_rate)
sampled_dR = avg_sample(dataRate, sample_rate)

#Uncomment to plot the sampled data
plt.title("Collected features with half-hour sample rate")
plt.xlabel("Time")

sampled_features = [sampled_nou, 
                    sampled_bits, 
                    sampled_pktNum, 
                    sampled_sigS, 
                    sampled_dR
                    ]
plot_colors = ["r",
               "b",
               "g",
               "y",
               "c"
               ]
for plot_iter in range(5):
    feature_plot(labels[plot_iter], sampled_features[plot_iter], plot_colors[plot_iter])
'''

#Uncomment to see Gaussian process
#Number of test samples
sample_start = 0
sample_end = 1000
#Window size
sample_window = 10
#Sample rate for the Gaussian Process
sample_rate_GP = 30


kernel1 = LK(sigma_0 = 1, sigma_0_bounds = (1e-1, 1e1))
kernel2 = CK(constant_value=1)
kernel3 = WK(0.1)
kernel = Sum(kernel1, kernel2)
kernel = Sum(kernel, kernel3)
#1e-1 for linear + constant, 1e-3 for RBF
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                              normalize_y=False, alpha=1e-1)
#print gp.get_params()['kernel']
training_samples = []
for z in range(len(labels)): #len(labels)
    total_samp, Xtr, Ytr, Xtst, Ycomp, Ytst, training_samples = GP_prep(training_data[z], test_data[z], sample_rate_GP, sample_start, sample_end, sample_window)
    
    #print(Xtr.shape, Ytr.shape, Xtst.shape, Ytst.shape)
    #Testing if it overfits
    Xtr_1 = [Xtr[i] for i in range(sample_window, sample_end)]
    
    #GP_ysample = gp.sample_y(Xtr_1, 1)
    try:
        gp.fit(Xtr, Ytr)
        print "marginal likelihood:", gp.log_marginal_likelihood()
        y_pred, y_sigma = gp.predict(Xtst, return_std=True)
        #print(y_pred.shape)
        
        result_time = [g+1 for g in range(sample_window, sample_end)]
        training_xaxis = [h for h in range(total_samp)]
        
        s = "training interval between "+str(result_time[0])+" and "+str(result_time[-1])+\
        " minutes\n window is "+str(sample_window)
        plt.xlabel(s=s)
        ylab = labels[z]
        plt.ylabel(ylab)
        plt.title(s="Training data with feature "+str(ylab))
        plt.plot(training_xaxis, training_samples, "g-", label="training")
        plt.legend()
        plt.show()
        
        GP_xlabel = "time interval between "+str(result_time[0])+" and "+str(result_time[-1])+\
        " minutes\n window is "+str(sample_window)
        plt.xlabel(s=GP_xlabel)
        GP_ylabel = labels[z]
        plt.ylabel(GP_ylabel)
        GP_dataplot_title = "Using "+str(gp.get_params()['kernel'])+" kernel\nwith "+str(total_samp)+" averaged training samples\nand "+str(sample_end)+\
        " averaged test samples"
        plt.title(s=GP_dataplot_title)
        
        #ploting data
        #plt.plot(result_time, GP_ysample, "c-", label= "kernel sample")
        plt.plot(result_time, y_pred.T[0], "c-", label="predicted")
        plt.plot(result_time, Ytst.T[0], "y-", label="real")
        plt.fill(np.concatenate([result_time, result_time[::-1]]),
                 np.concatenate([y_pred-1.96*y_sigma,
                                (y_pred+1.96*y_sigma)[::-1]]),
                 alpha=.5, fc='b', ec='none')
        plt.legend()
        plt.show()
        
    except:
        print "Feature "+str(labels[z])+" did not work!"
        continue
