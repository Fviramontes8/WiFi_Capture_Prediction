#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Created on Fri Sep 01 11:06:34 2017

@author Francisco Viramontes
Todo: All three weeks in one go to predict one minute ahead
'''
import DatabaseConnector as dc
import numpy as np
np.set_printoptions(threshold=np.nan)
from numpy.fft import fft, fftshift

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as LK, WhiteKernel as WK
from sklearn.gaussian_process.kernels import ConstantKernel as CK, Sum

from scipy import signal

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
    print(sample_arr, sample_size)
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
def GP_prep_old(train, test, avg_samp, sub_samp_begin, sub_samp_end, window):
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

def read_5ghz_day(table_name):
    db = dc.DatabaseConnect()

    #print(db.getTableNames())

    #create_table_name = "first_test3"
    #db.createDataTable_5ghz(create_table_name)

    db.test_connect(database_name="postgres", username="postgres", host_name="18.221.41.211", password_name="Cerculsihr4T")
    data = db.readTable(table_name)
    db.disconnect()

    t_stamps = []
    num_of_users = []
    bits = []
    '''
    pktNum = []
    sigStrength = []
    data_rate = []
    phyA_bits = []
    phyN_bits= []
    '''

    for db_iter in sorted(data, key=lambda dummy:dummy[1]):
        t_stamps.append(db_iter[1])
        num_of_users.append(db_iter[2])
        bits.append(db_iter[3])
        '''
        pktNum.append(db_iter[4])
        sigStrength.append(db_iter[5])
        data_rate.append(db_iter[6])
        phyA_bits.append(db_iter[7])
        phyN_bits.append(db_iter[8])
        '''

    return_data = [t_stamps,
                   num_of_users,
                   bits]
    '''
                   pktNum,
                   sigStrength,
                   data_rate,
                   phyA_bits,
                   phyN_bits]
    '''
    return return_data

def butterfilter(input_arr, title, sampling=60):
    z = (0.9/4) / sampling
    begin_cutoff = 500
    b, a = signal.butter(6, z, 'low')
    xf = signal.lfilter(b, a, input_arr)
    plt.plot(input_arr[begin_cutoff:])
    plt.plot(xf[begin_cutoff:])
    xf_copy = np.array(xf).copy()
    xs = xf_copy[1::sampling]
    x_axis_xs = np.array([i for i in range(len(xf))])
    x_axis_xs = x_axis_xs[::sampling]
    plt.title("Time series and filtered data for "+title)
    plt.show()
    print(len(xs))
    return xs
    '''
    plt.plot(x_axis_xs[int(begin_cutoff/sampling):int(len(xf)/sampling)-1], xs[int(begin_cutoff/sampling):int(len(xf)/sampling)-1], 'c')
    plt.title("Filtered data of "+title)
    plt.show()
    '''

if __name__ == '__main__':

    '''
    db = dc.DatabaseConnect()
    db.test_connect(database_name="postgres", username="postgres", host_name="18.221.41.211", password_name="Cerculsihr4T")
    db.createDataTable_5ghz("5pi_sun4")
    db.createDataTable_5ghz("5pi_mon4")
    db.createDataTable_5ghz("5pi_tues4")
    db.createDataTable_5ghz("5pi_wed4")
    db.createDataTable_5ghz("5pi_thurs4")
    db.createDataTable_5ghz("5pi_fri4")
    db.createDataTable_5ghz("5pi_sat4")
    print(db.getTableNames())
    db.disconnect()
    '''


    #sun1 = read_5ghz_day("5pi_sun")
    sun2 = read_5ghz_day("5pi_sun2")

    mon1 = read_5ghz_day("5pi_mon")
    #mon2 = read_5ghz_day("5pi_mon2")

    #tues1 = read_5ghz_day("5pi_tues")
    #tues2 = read_5ghz_day("5pi_tues2")

    #wed1 = read_5ghz_day("5pi_wed")
    #wed2 = read_5ghz_day("5pi_wed2")

    #thurs1 = read_5ghz_day("5pi_thurs")
    #thurs2 = read_5ghz_day("5pi_thurs")

    #fri = read_5ghz_day("5pi_fri")
    #fri2 = read_5ghz_day("5pi_fri2")

    #sat = read_5ghz_day("5pi_sat")
    #sat2 = read_5ghz_day("5pi_sat2")


    #wed3 = read_5ghz_day("5pi_wed3")
    #f_wed3 = []

    f_sun1 = []
    f_sun2 = []
    f_mon1 = []
    f_mon2 = []

    f_tues1 = []

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
    '''
    for q in range(len(sun1)-1):
        f_sun1.append(butterfilter(sun1[q+1], labels_5ghz[q]))
    '''
    for p in range(len(sun2)-1):
        f_sun2.append(butterfilter(sun2[p+1], labels_5ghz[p]))

    for n in range(len(mon1)-1):
        f_mon1.append(butterfilter(mon1[n+1], labels_5ghz[n]))
    '''
    for m in range(len(mon2)-1):
        f_mon2.append(butterfilter(mon2[m+1], labels_5ghz[m]))
    '''
    '''
    for l in range(len(wed3)-1):
        f_wed3.append(butterfilter(wed3[l+1], labels_5ghz[l]))
    '''

    #print(len(f_sun1), len(f_sun2), len(f_mon1), len(f_mon2))
    #print(len(f_sun1[0]), len(f_sun1[1]))
    #sun1_tstamp = sun1[0][0::60]

    #sun2_tstamp = sun2[0][0::60]
    #mon1_tstamp = mon1[0][0::60]
    #mon2_tstamp = mon2[0][0::60]


    '''#Uncomment to see sampled features
    sample_rate = 1800

    sampled_nou = avg_sample(nou, sample_rate)
    sampled_bits = avg_sample(bits, sample_rate)
    sampled_pktNum = avg_sample(pktNum, sample_rate)
    sampled_sigS = avg_sample(sigS, sample_rate)
    sampled_dR = avg_sample(dataRate, sample_rate)
    print(sampled_nou.shape)
    print(sampled_bits.shape)
    print(sampled_pktNum.shape)
    print(sampled_sigS)
    print(sampled_dR)
    '''

    '''
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

    '''#Uncomment to see Gaussian process
    #Number of test samples
    sample_start = 0
    sample_end = 100
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
            print("marginal likelihood:", gp.log_marginal_likelihood())
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
            print("Feature "+str(labels[z])+" did not work!")
            continue
        '''