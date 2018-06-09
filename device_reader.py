#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
@author Francisco Viramontes

Description: Reads mac/ip addresses from specific tables from a database.

Input: String of a table name in database, needs to have columns [key, mac_addr, ip_addr]

Output: Contents of the table
'''

import DatabaseConnector as dc

table_name = "ip"

db = dc.DatabaseConnect()
db.connect()
table_contents = db.readTable(table_name)
db.disconnect()

#print(table_contents)


device_details = {} #mac address is key, ip address is value

for i in sorted(table_contents, key=lambda hello: hello[0]):
    device_details[i[1]] = i[2]

#print(device_details)
for j in device_details:
    print(str(j) + " -> " + str(device_details[j]))
