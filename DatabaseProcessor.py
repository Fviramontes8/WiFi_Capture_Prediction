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

'''
(Key, ts, nou, bits, pkt_num,
		sigs, dr, phya, phyn)
'''
def select_feature(feature_name):
	assert(type(feature_name) == str)
	feature_str = feature_name.lower()
	if (feature_str == "timestamp") | (feature_str == "ts"):
		return 1
	elif (feature_str == "nou") | (feature_str == "num_users"):
		return 2
	elif feature_str == "bits":
		return 3
	elif (feature_str == "pkt_num") | (feature_str == "packet_number"):
		return 4
	elif (feature_str == "sigs") | (feature_str == "signal_strength"):
		return 5
	elif (feature_str == "dr") | (feature_str == "data_rate"):
		return 6
	elif feature_str == "phya":
		return 7
	elif feature_str == "phyn":
		return 8
	else:
		return 0
		
		
def pull_week_data(num_weeks, feature_name):
	days_of_week = ["mon",
				 "tues",
				 "wed",
				 "thurs",
				 "fri"
		]
	feature_index = select_feature(feature_name)
	week_data = []
	db = dc.DatabaseConnect()
	#old_len = 0
	
	for week_num in range(2, num_weeks+1):
		for day in days_of_week:
			table_name = "5pi_"+str(day)+str(week_num)
			db.connect()
			db_data = db.readTable(table_name)
			db.disconnect()
			for db_iter in sorted(db_data, key=lambda dummy_arr:dummy_arr[1]):
				week_data.append(db_iter[feature_index])
			'''
			while (week_data[old_len] < 1):
				del week_data[old_len]
			old_len = len(week_data)
			'''
		
	return week_data