#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:50:08 2020

Packages needed: scikit-learn, psycopg2, numpy, scipy

@author: frankie
From: https://github.com/fviramontes8/Wifi_Capture_Prediction
Depends on file: Signal Processor.py, DatabaseConnector.py
"""

#Package to interface with AWS database
import DatabaseConnector as dc

#Private signal processor/sampler
import SignalProcessor as sp

import numpy as np

def print_table_names():
	database = dc.DatabaseConnect()
	database.connect()
	table_names = database.getTableNames()
	print(table_names)
	database.disconnect()

def create_table(table_name):
	database = dc.DatabaseConnect()
	database.connect()
	database.createDataTable_5ghz(table_name)
	database.disconnect()

def del_table(table_name):
	db = dc.DatabaseConnect()
	db.connect()
	db.drop_table(table_name)
	db.disconnect()


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
	#print("Next key is:", db.getNextKey(table_name))
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
					bits
					]
	return return_data

def week_bits_prep(day, start_week, end_week, sample_rate, sample_rate2=None):
	training_data = []
	labels_5ghz = ["Number of users",
       				"Bits"
       				]

	for week in range(start_week, end_week+1):
		table_name = "5pi_"+str(day)+str(week)
		day_data = read_5ghz_day(table_name)

		while(day_data[2][0] < 1):
				del day_data[2][0]
		filtered_data = sp.butterfilter(day_data[2], labels_5ghz[1], table_name)
		sampled_data = sp.sub_sample(filtered_data, labels_5ghz[1], table_name, sample_rate)
		if(sample_rate2):
			sampled_data = sp.sub_sample(sampled_data, labels_5ghz[1], table_name, sample_rate2)
		training_data.extend(sampled_data)

	training_data = np.array(training_data)
	return training_data

def day_bits_prep(days_of_week, num_of_weeks, sample_rate, sample_rate2=None):
	training_data = []
	labels_5ghz = ["Number of users",
       				"Bits"
       				]
	for week_num in range(2, num_of_weeks+1):
		for day in days_of_week:
			table_name = "5pi_"+str(day)+str(week_num)
			day_data = read_5ghz_day(table_name)

			while(day_data[2][0] < 1):
				del day_data[2][0]
			filtered_data = sp.butterfilter(day_data[2], labels_5ghz[1], table_name)
			sampled_data = sp.sub_sample(filtered_data, labels_5ghz[1], table_name, sample_rate)
			if(sample_rate2):
				sampled_data = sp.sub_sample(sampled_data, labels_5ghz[1], table_name, sample_rate2)
			training_data.extend(sampled_data)

	training_data = np.array(training_data)
	return training_data