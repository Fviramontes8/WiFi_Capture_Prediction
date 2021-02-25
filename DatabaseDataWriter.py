# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:58:09 2021

@author: Frankie
"""
import numpy as np
from os import path

# Private database processor
import DatabaseProcessor as dbp
# Private signal processor/sampler
import SignalProcessor as sp
# Private plot functions
import PlotUtils as pu

if __name__ == "__main__":
	data_folder = "data/"
	days_of_week = ["mon",
				 "tues",
				 "wed",
				 "thurs",
				 "fri"
				 ]
	day = "thurs"
	labels_5ghz = ["Number of users",
       				"Bits"
       				]
	
	begin_week = 2
	end_week = 14
	init_sample_rate = 60
	second_sample_rate = 60
	test_day = days_of_week[0]
	test_week=15
	total_weeks=5
	
	# Needs to be called for a new batch of data
	if not path.exists(data_folder+"tr_bits_"+str(total_weeks)+"weeks_hoursample.npy"):
		bits = dbp.day_data_prep(days_of_week, total_weeks, init_sample_rate, second_sample_rate)
		print(len(bits))
		np.save(data_folder+"tr_bits_"+str(total_weeks)+"weeks_hoursample", bits)
		bits = sp.std_normalization(bits)
		np.save(data_folder+"tr_bits_"+str(total_weeks)+"weeks_hoursample_normalized", bits)
		
		bits_title = str(total_weeks)+" weeks of bits"
		bits_xtitle = "Time (seconds)"
		bits_ytitle = "Bits"
		pu.general_plot(bits, bits_title, bits_xtitle, bits_ytitle)
	
	
	if not path.exists(data_folder+"tst_bits_week"+str(test_week)+str(test_day)+"_hoursample.npy"):
		Xtst = dbp.week_data_prep(test_day, test_week, test_week, init_sample_rate, second_sample_rate)
		np.save(data_folder+"tst_bits_week"+str(test_week)+str(test_day)+"_hoursample", bits)
		print(len(Xtst))
		Xtst = sp.std_normalization(Xtst)
		np.save(data_folder+"tst_bits_week"+str(test_week)+str(test_day)+"_hoursample_normalized", Xtst)