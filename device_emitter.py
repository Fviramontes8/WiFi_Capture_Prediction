# -*- coding: utf-8 -*-
'''
Author: Fviramontes8

Description: This program reads two files: ip.txt and mac.txt, it receives
    the device's ip and mac address from the files and checks them in a
    PostgreSQl database. If the mac address is not in the database, the
    program uploads both addresses into the database. If the mac address
    exists, it checks if the ip address matches what is in the database.
    If there is a match, nothing will happen, else, it will update the 
    database.
    
Input: 'ip.txt', 'mac.txt' must contain ONE ip/mac address respectively.

Output: None to the user's terminal, updates PostgreSQL database
'''

def txt_reader(input_file):
    '''Reads a text file and returns its contents as one string'''
    txt_obj = open(input_file, "r")
    return_addr= txt_obj.read()
    txt_obj.close()
    
    return_addr = return_addr.strip("\n")
    return_addr = return_addr.strip()
    
    return return_addr

def data_upload(table_name, input_ip, input_mac):
    '''Uploads data using writeDeviceData() from DatabaseConnector'''
    db.connect()
    key = db.getNextKey(table_name)
    key += 1
    #print("Next key: " + str(key))
    pi_upload = [key, input_mac, input_ip]
    db.writeDeviceData(table_name, pi_upload)
    db.disconnect()
    

import DatabaseConnector as dc

table_name = "ip"
db = dc.DatabaseConnect()
db.connect()
table_contents = db.readTable(table_name)
db.disconnect()

#print(table_contents[0][0])
pi_details = {}
for i in table_contents:
    pi_details[i[0]] = [i[1], i[2]]
#print(pi_details)

#text_object = open("ip.txt", "r")
#print(text_object.read())
#ip_address = text_object.read()
#text_object.close()
#Cleans up what is parsed in the text file
#ip_address = ip_address.strip("\n")
#ip_address = ip_address.strip()
#print("This computer's IP address: " + ip_address)
ip_txt = "ip.txt"
ip_addr = txt_reader(ip_txt)

mac_txt = "mac.txt"
mac_addr = txt_reader(mac_txt)
#text_object = open("mac.txt", "r")
#mac_address = text_object.read()
#text_object.close()
#mac_address = mac_address.strip("\n")
#print("This computer's MAC address: " + mac_address)

for pi_iterator in range(len(pi_details)):
    database_ip = pi_details[pi_iterator+1][1]
    #print("Database: " + database_ip)
    if(database_ip == ip_addr):
        ip_check = 0
        print("There is an IP address, here is the key: " +str(pi_iterator+1))
    else:
        #print("There is no match for IP!")
        ip_check = 1
    #print("ip_check: " + str(ip_check))

mac_check = 0

if(ip_check):
    for pi_iterator in range(len(pi_details)):
        database_mac = pi_details[pi_iterator+1][0]
        #print("Database: " + database_mac)
        if(database_mac == mac_addr):
            mac_check = 0
            print("There is a MAC address, here is the key: " +str(pi_iterator+1))
            #db.deleteIPData(pi_iterator+1)
        else:
            mac_check = 1
            #print("There is no match for MAC!")
        #print("mac_check: " + str(mac_check))

if(mac_check):
    print("This device is not on the database!")
    data_upload(table_name, ip_addr, mac_addr)
