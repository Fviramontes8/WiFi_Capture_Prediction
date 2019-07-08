# -*- coding: utf-8 -*-
'''
@author: Seth Decker, Francisco Viramontes

Description: This code interfaces to a PostgreSQL database. It can build tables,
read tables, and write to tables.


Example:

    #Read data table
>>> import DatabaseConnect as dc

    database = dc.DatabaseConnect()

    database.connect()

    print(database.readDataTable())

    database.disconnect()


    #Write data table
>>> import DatabaseConnect as dc

    database = dc.DatabaseConnect()

    data_table_name = "example_table"

    data_list = []
    data_list.append((1, 2, "dummy", "dummy", "dummy", "dummy", 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))
    data_list.append((2, 3, "dummy", "dummy", "dummy", "dummy", 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25))

    database.writeData(data_table_name, data_list)

    database.disconnect()

'''

from configparser import ConfigParser
import os
import psycopg2
from psycopg2 import sql
#import pandas as pd

class DatabaseConnect(object):
    '''Database Object to interact with PostgreSQL database'''
    def __init__(self):
        '''Default Constructor for the class'''
        self.conn = None
        ## @var  conn
        #Variable that determines if connection has been established with database

        self.data_2ghz_query = "(Key, ts, nou, bits, pkt_num, sigs, dr, phyb, phyg, phyn) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.data_5ghz_query = "(Key, ts, nou, bits, pkt_num, sigs, dr, phya, phyn) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        ## @var data_2ghz_query
        #Variable that is used for tables that contain wifi data on the 2ghz spectrum, query: (Key, ts, nou, bits, pkt_num, sigs, dr, phyb, phyg, phyn)

    def _checkConnection(self):
        '''Checks connection with PostgreSQL database'''
        if self.conn is None:
            print("No connection established. Use obj.connect() to connect to database.")
            return False
        else:
            return True
    def drop_table(self, table_name):
        '''Deletes table_name from database'''
        if self._checkConnection():
            cur = self.conn.cursor()
            query = sql.SQL("DROP TABLE {}").format(sql.Identifier(table_name))
            cur.execute(query)
            self.conn.commit()

    def _config(self, filename='database.ini', section='postgresql'):
        '''Reads 'database.ini' file from current directory to get initialization values to connect to database.
        An example of an valid 'database.ini' file is:
               ; Specifies what kind of database this file is going to work with
               [postgresql]

                ;Specifies the user credentials and IP address to connect to
                host= 127.0.0.1
                database= table_data
                user= example_username
                password= dont_panic
            '''
        # create a parser
        parser = ConfigParser()
        # read config file
        if os.path.exists(filename):
            parser.read(filename)

            # get section, default to postgresql
            db = {}
            if parser.has_section(section):
                params = parser.items(section)
                for param in params:
                    db[param[0]] = param[1]
            else:
                print("expection called")
                raise Exception('Section {0} not found in the {1} file'.format(section, filename))
            return db
        else:
            print("Configuration file does not exist")
            return None

    def tableExists(self, cursor, table_name):
        '''Checks if table exists in database'''
        cursor.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
        return cursor.fetchone()[0]

    ## Public Functions
    def getNextKey(self, table_name):
        '''Gets the next key needed to upload to the database, insures that there are no key conflicts'''
        if self._checkConnection():
            cur = self.conn.cursor()
            query = sql.SQL("SELECT MAX(Key) from {}").format(sql.Identifier(table_name))
            cur.execute(query)
            latest_key = cur.fetchall()
            res_list = [x[0] for x in latest_key]
            return res_list[0]

    def readTable(self, table_name):
        '''Returns the entire content of table_name'''
        if self._checkConnection():
            cur = self.conn.cursor()
            query = sql.SQL("select * from {} as a").format(sql.Identifier(table_name))
            cur.execute(query)
            print("Data retrieved from table: " + table_name)

            output = cur.fetchall()

            return output
        return

    def _writeData(self, table_name, query, new_data):
        #print(query)
        #print(table_name)
        #print(new_data)

        if self._checkConnection():
            cur = self.conn.cursor()

            sql_query = sql.SQL("INSERT INTO {} " + query).format(sql.Identifier(table_name))

            print(sql_query)

            # need to discriminate single and multiple data
            cur.execute(sql_query, new_data)
            #cur.executemany(sql_query, new_data)

            self.conn.commit()

    def writeData_2ghz(self, table_name, new_table_data):
        '''Writes data to table of PostgreSQL database, input must be in this from: Key, ts, nou, bits, pkt_num, sigs, dr, phyb, phyg, phyn. Where all values are integers'''
        self._writeData(table_name, self.data_2ghz_query, new_table_data)
    def writeData_5ghz(self, table_name, new_table_data):
        '''Writes data to table of PostgreSQL database, input must be in this from: Key, ts, nou, bits, pkt_num, sigs, dr, phya, phyn. Where all values are integers'''
        self._writeData(table_name, self.data_5ghz_query, new_table_data)
    #'''
    def writeDeviceData(self, table_name, table_data):
        '''This function takes a list of [key, mac addr, ip addr] specified by table_data and adds it to database specified by table_name'''
        table_query = "(Key , mac, ip) VALUES (%s, %s, %s)"
        if self._checkConnection():
            cur = self.conn.cursor()
            query = sql.SQL("INSERT INTO {} " + table_query).format(sql.Identifier(table_name))
            cur.execute(query, table_data)
            self.conn.commit()
    #'''
    def delete5Data(self, key, table_name):
        self._deleteData(key, table_name)

    def _deleteData(self, key, table_name):
        '''Deletes data specified by key and table_name'''
        if self._checkConnection():
            cur = self.conn.cursor()
            query = sql.SQL("delete from {} where Key={}").format(sql.Identifier(table_name), sql.Identifier(key))
            cur.execute(query)

    def test_connect(self, database_name, username, host_name, password_name):
        self.conn = psycopg2.connect(database=database_name, user=username, host=host_name, password=password_name)

    def connect(self):
        '''Tries to establish a connection with the database'''
        try:
            params = self._config()

            if params is None:
                print("Unable to connect to the database! No Params.")
                return

            self.conn = psycopg2.connect(database=params["database"], user=params['user'], host=params["host"], password=params["password"])
            print("Connected.")
        except:
            print("Unable to connect to the database!")

    def disconnect(self):
        '''Disconnects from the database'''
        if self.conn is not None:
            self.conn.close()
        print("Disconnected.")


    def createDataTable_2ghz(self, table_name):
        '''Creates table specified by table_name with columns: (Key, ts, nou, bits, pkt_num, sigs, dr, phyb, phyg, phyn). They are all integer values.
        This is a template for statistics that we available in the 2.4GHz spectrum of Wi-Fi'''
        ##@var table_name
        #Name of the table the user wants to get data from in database
        if self._checkConnection():
            #moving key->value to post processing
            args = "(Key INT PRIMARY KEY, ts INT, nou INT, bits INT, pkt_num INT, sigS INT, dr INT, phyb INT, phyg INT, phyn INT)"
            query = sql.SQL("CREATE TABLE {} " + args).format(sql.Identifier(table_name))
            self.conn.cursor().execute(query)
            self.conn.commit()

    def createDataTable_5ghz(self, table_name):
        '''Creates table specified by table_name with columns: (Key, ts, nou, bits, pkt_num, sigs, dr, phyb, phyg, phyn). They are all integer values.
        This is a template for statistics that we available in the 5GHz spectrum of Wi-Fi'''
        ##@var table_name
        #Name of the table the user wants to get data from in database
        if self._checkConnection():
            #moving key->value to post processing
            args = "(Key INT PRIMARY KEY, ts INT, nou INT, bits INT, pkt_num INT, sigS INT, dr INT, phya INT, phyn INT)"
            query = sql.SQL("CREATE TABLE {} " + args).format(sql.Identifier(table_name))
            self.conn.cursor().execute(query)
            self.conn.commit()

    def getTableNames(self):
        '''Returns a list of table names in the database'''
        #In order to see tables names the user should print it
        #print DatabaseConnect.getTableNames()
        if self._checkConnection():
            cur = self.conn.cursor()

            cur.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
            out = cur.fetchall()
            table_names = [x[0] for x in out]
            return table_names
