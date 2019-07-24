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
np.set_printoptions(threshold=np.nan)

#For ploting data
import matplotlib.pyplot as plt

#For use of the butterworth and Savgol filters
from scipy import signal

#Machine Learning package for the Gaussian Process Regressor
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
        return 0

def GP_Prep(training, testing, window):
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
    Xtr = np.atleast_2d([grab_nz(training, m, n) for m, n in zip(range(training.shape[0]), range(window, training.shape[0]))])
    Ytr = np.atleast_2d([[training[i] for i in range(window, training.shape[0])]]).T
    Xtst = np.atleast_2d([grab_nz(testing, m, n) for m, n in zip(range(testing.shape[0]), range(window, testing.shape[0]))])
    Ycomp = np.atleast_2d([testing[i] for i in range(window, testing.shape[0])]).T

    return Xtr, Ytr, Xtst, Ycomp

def print_gp(pred, sigma, compare, feature, day, window):
    sigma = 0.98#1.96
    prediction_time= [p+1 for p in range(len(pred))]
    plt.plot(prediction_time, pred, "c-", label="GP Prediction")
    plt.plot(prediction_time, compare, "y.", label="Validation data")
    plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
                     np.concatenate([y_pred-sigma*y_sigma,
                                    (y_pred+sigma*y_sigma)[::-1]]),
                     alpha=.5, fc='b', ec='none')
    plt.legend()
    plt.title("Gaussian Process Prediction with 6th order Butterworth filter,\nPredicting "+day+"day\nWith window of "+str(window))
    plt.xlabel("Time (Hours)")
    plt.ylabel(feature+" (predicted)")
    plt.show()

    sigma *= 2
    plt.plot(prediction_time, pred, "c-", label="GP Prediction")
    plt.plot(prediction_time, compare, "y.", label="Validation data")
    plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
                     np.concatenate([y_pred-sigma*y_sigma,
                                    (y_pred+sigma*y_sigma)[::-1]]),
                     alpha=.5, fc='b', ec='none')
    plt.legend()
    plt.title("Gaussian Process Prediction with 6th order Butterworth filter,\nPredicting "+day+"day\nWith window of "+str(window))
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
    print("Next key is:", db.getNextKey(table_name))
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
                   bits]
    return return_data

def butterfilter(input_arr, title, sampling=60):
    '''Input:
        A list that can be represented as a time series that is the feature
            desired to be filtered (input_arr)
        A string that describes the filter (title)
        A sampling frequency (sampling) [the default value is 60 to sample
            the data into minutes chunks]
        Output:
        A list of filtered data points (1/sampling) of original size
    '''
    z = (0.9/4) / sampling
    print("Length: ", len(input_arr))
    print("Z: ", z)
    begin_cutoff = 0
    b, a = signal.butter(6, z, 'low')
    xf = signal.lfilter(b, a, input_arr)
    plt.plot(input_arr[begin_cutoff:], label="Original Data")
    plt.plot(xf[begin_cutoff:], label="Filtered Data")
    return xf

def custom_sample(xf, title, sampling=60):
    xf_copy = np.array(xf).copy()
    xs = xf_copy[1::sampling]
    x_axis_xs = np.array([i for i in range(len(xf))])
    x_axis_xs = x_axis_xs[::sampling]
    plt.title("Time series and filtered data for "+title)
    plt.ylabel(title)
    plt.xlabel("Time of day (seconds)")
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
        kernel1 = LK(sigma_0 = 1, sigma_0_bounds = (1e-1, 1e1))
        kernel2 = CK(constant_value=1)
        kernel3 = WK(0.1)
        kernel = Sum(kernel1, kernel2)
        kernel = Sum(kernel, kernel3)
    elif kernel_str == "RBF":
        kernel1 = RBF(length_scale=5, length_scale_bounds=(1e-1, 1e1))
        kernel2 = CK(constant_value=1)
        kernel3 = WK(0.1)
        kernel = Sum(kernel1, kernel2)
        kernel = Sum(kernel, kernel3)
    else:
        print("Not a valid kernel name, defaulting to ", end="")
        kernel1 = LK(sigma_0 = 1, sigma_0_bounds = (1e-1, 1e1))
        kernel2 = CK(constant_value=1)
        kernel3 = WK(0.1)
        kernel = Sum(kernel1, kernel2)
        kernel = Sum(kernel, kernel3)
        print(kernel)

    #print(kernel)
    return kernel


if __name__ == '__main__':
    #Reading data from database
    #Good days:
    # mon(86367), mon2(41297), tues2(30813), wed(54312), wed2(43381),
    # thurs(31189), thurs2(79178), fri(50628), fri2(82396), sat(86339)
    # sat2(71554), sun(75978), sun2(86535)
    day = "fri"
    mon = read_5ghz_day("5pi_"+day)
    #mon2 = read_5ghz_day("5pi_mon2")
    #tues = read_5ghz_day("5pi_tues2")
    #sun = read_5ghz_day("5pi_sun2")

#    test = read_5ghz_day("5pi_mon3")
#    print(len(test[1]))

#    '''
    #1-D string array of features to examine
    labels_5ghz = ["Number of users",
                   "Bits"
                   ]

    '''
    #Plotting the original data
    plt.plot(mon[2], "r")
    plt.ylabel("Bits (Training)")
    plt.xlabel("Time of day (seconds)")
    plt.title("Bits sent through the network")
    plt.show()

    plt.plot(sun[2], "r")
    plt.ylabel("Bits (Testing)")
    plt.xlabel("Time of day (seconds)")
    plt.title("Bits sent through the network")
    plt.show()
    '''

    #Putting the data through a butterworth filter
    #butter_nou_tr1 = butterfilter(mon[1], labels_5ghz[0])
    butter_bits_tr1 = butterfilter(mon[2], labels_5ghz[1])
    butter_bits_tr = butter_bits_tr1

    for zed in range(2, 8):
        try:
            temp_mon = read_5ghz_day("5pi_"+day+str(zed))

            while(temp_mon[2][0] < 1):
                del temp_mon[2][0]

            temp_bits_tr = butterfilter(temp_mon[2], labels_5ghz[1], 60)
            butter_bits_tr = list(butter_bits_tr) + list(temp_bits_tr)
            butter_bits_tr = np.array(butter_bits_tr)
        except:
            print("Could not read 5pi_"+day+str(zed))

    '''
    butter_bits_tr2 = butterfilter(mon2[2], labels_5ghz[1])
    butter_bits_tr = list(butter_bits_tr) + list(butter_bits_tr2)
    butter_bits_tr = np.array(butter_bits_tr)

    butter_bits_tr3 = butterfilter(tues[2], labels_5ghz[1])
    butter_bits_tr = list(butter_bits_tr) + list(butter_bits_tr3)
    butter_bits_tr = np.array(butter_bits_tr)
    '''

    plt.plot(butter_bits_tr, "m")
    plt.ylabel("Bits sent")
    plt.xlabel("Time in minutes")
    plt.title("7 days of Monday bits sent")
    plt.show()

    butter_bits_tr = butterfilter(butter_bits_tr, labels_5ghz[1], 6)
    plt.plot(butter_bits_tr, "m")
    plt.ylabel("Bits sent")
    plt.xlabel("Time in hours")
    plt.title("7 days of Monday bits sent")
    plt.show()


    mon_tst = read_5ghz_day("5pi_"+day+"8")
    while(mon_tst[2][0] < 1):
            del mon_tst[2][0]
    butter_bits_tst = butterfilter(mon_tst[2], labels_5ghz[1])

    plt.plot(butter_bits_tst, "g")
    plt.ylabel("Bits sent")
    plt.xlabel("Time in minutes")
    plt.title("Testing data of bits sent")
    plt.show()

    butter_bits_tst = butterfilter(butter_bits_tst, labels_5ghz[1], 60)
    plt.plot(butter_bits_tst, "g")
    plt.ylabel("Bits sent")
    plt.xlabel("Time in hours")
    plt.title("Testing data of bits sent")
    plt.show()


    #Testing is 5pi_mon8

    #butter_nou_tst = butterfilter(sun[1], labels_5ghz[0])
    #butter_bits_tst = butterfilter(sun[2], labels_5ghz[1])
#    '''

    '''
    #Split data into a training an validaiton set with a ratio of 90:10
    butter_nou_tr = butter_nou[:int(len(butter_nou)*0.9):]
    butter_nou_tst = butter_nou[int(len(butter_nou)*0.9)::]

    xaxis1 = [i for i in range(len(butter_nou_tr))]
    plt.plot(xaxis1, butter_nou_tr, "c", label="Training")
    xaxis2 = [j for j in range(len(butter_nou_tr), len(butter_nou_tr)+len(butter_nou_tst))]
    plt.plot(xaxis2, butter_nou_tst, "g", label="Validation")
    plt.legend()
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Number of Users")
    plt.title(s="Butterworth filtered data for number of users")
    plt.show()
    '''
#    '''
    #Declaration of the Gaussian Process Regressor with its kernel parameters
    kernel = kernel_select("linear")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                                  normalize_y=False)#, alpha=1e-3)
    window = 10
    #Transforming the input data so that it can be used in the Gaussian Process
    Xtr, Ytr, Xtst, Ycomp = GP_Prep(butter_bits_tr, butter_bits_tst, window)

    print(Xtr.shape)
    print(Xtst.shape)

    print("Training the Gaussian Process...")
    gp.fit(Xtr, Ytr)
    print("\tMarginal likelihood:", gp.log_marginal_likelihood())


    #Plotting prediction against mon8
    y_pred, y_sigma = gp.predict(Xtst, return_std=True)
    print("Chi-squared test against training data: ", gp.score(Xtr,Ytr))
    print("Chi-squared test against real data: ", gp.score(Xtst,Ycomp))
    print("MAPE between acutal and estimated:", mape_test(Ycomp, y_pred) * 100)
    print_gp(y_pred, y_sigma, Ycomp, "Bits", day, str(window)+"\nMAPE: "+str(mape_test(Ycomp, y_pred) * 100))

    '''
    #Plotting prediction against mon1
    Xtr, Ytr, Xtst, Ycomp = GP_Prep(butter_bits_tr, butter_bits_tr1, window)
    y_pred, y_sigma = gp.predict(Xtst, return_std=True)
    print_gp(y_pred, y_sigma, Ycomp, "Bits")
    print(gp.score(Xtr,Ytr))
    print(gp.score(Xtst,Ycomp))

    #Plotting prediction against mon2
    Xtr, Ytr, Xtst, Ycomp = GP_Prep(butter_bits_tr, butter_bits_tr2, window)
    y_pred, y_sigma = gp.predict(Xtst, return_std=True)
    print_gp(y_pred, y_sigma, Ycomp, "Bits")
    print(gp.score(Xtr,Ytr))
    print(gp.score(Xtst,Ycomp))


    #Plotting prediction against tues2
    Xtr, Ytr, Xtst, Ycomp = GP_Prep(butter_bits_tr, butter_bits_tr3, window)
    y_pred, y_sigma = gp.predict(Xtst, return_std=True)
    print_gp(y_pred, y_sigma, Ycomp, "Bits")
    print(gp.score(Xtr,Ytr))
    print(gp.score(Xtst,Ycomp))
#    '''

    '''
    #Training data is a time series
    #kernel1 = RBF(length_scale=1, length_scale_bounds=(1e-1, 1e1))
    kernel1 = LK(sigma_0 = 0.1, sigma_0_bounds = (1e-3, 1e3))
    kernel2 = CK(constant_value=5)
    #kernel3 = WK(1)
    kernel = Sum(kernel1, kernel2)
    #kernel = Sum(kernel, kernel3)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                                  normalize_y=False, alpha=1e-1)
    print("Training the Gaussian Process...\n")
    print(butter_bits_tr.shape, butter_bits_tst.shape)
    real_xtr = [some_q+1 for some_q in range(len(butter_bits_tr))]
    real_xtr = np.atleast_2d(real_xtr).T

    plt.plot(real_xtr, butter_bits_tr)
    plt.plot(butter_bits_tst)
    plt.show()

    gp.fit(real_xtr, butter_bits_tr)
    y_pred, y_sigma = gp.predict(np.atleast_2d(butter_bits_tst).T, return_std=True)
    prediction_time= [p+1 for p in range(len(y_pred))]
    plt.plot(prediction_time, y_pred, "c-", label="GP Prediction")
    plt.plot(butter_bits_tst, "y.", label="Validation data")
    plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
                     np.concatenate([y_pred-1.96*y_sigma,
                                    (y_pred+1.96*y_sigma)[::-1]]),
                                     alpha=.5, fc='b', ec='none')
    plt.legend()
    plt.title("Gaussian Process Prediction with 6th order Butterworth filter,\nTesting is a time series")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Bits (predicted)")
    plt.show()
    '''

    '''
    #Savgol filtering
    savgol_bits_tr = savgol(mon[2], labels_5ghz[1])
    savgol_bits_tst = savgol(sun[2], labels_5ghz[1])

    Xtr, Ytr, Xtst, Ycomp = GP_Prep(savgol_bits_tr, savgol_bits_tst, 10)

    print("Training the Gaussian Process...\n")
    gp.fit(Xtr, Ytr)
    print("\tMarginal likelihood:", gp.log_marginal_likelihood())
    y_pred, y_sigma = gp.predict(Xtst, return_std=True)

    #Plotting prediction
    prediction_time= [p+1 for p in range(len(y_pred))]
    plt.plot(prediction_time, y_pred, "c-", label="GP Prediction")
    plt.plot(prediction_time, Ycomp, "y.", label="Validation data")
    plt.fill(np.concatenate([prediction_time, prediction_time[::-1]]),
                     np.concatenate([y_pred-1.96*y_sigma,
                                    (y_pred+1.96*y_sigma)[::-1]]),
                     alpha=.5, fc='b', ec='none')
    plt.legend()
    plt.title("Gaussian Process Prediction with Savitzky-Golay filter")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Bits (predicted)")
    plt.show()
    '''