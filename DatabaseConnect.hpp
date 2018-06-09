/************************************************************************
 * @file DatabaseConnect.hpp
 * @author Seth Decker, Francisco Viramontes
 * 
 * @brief Header file for DatabaseConnect.cpp and parser5.cpp
 * 
 * Description: This file is the declaration of the database class used to 
 *  interact between the database and the pcap parser
 ************************************************************************/
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <postgresql/libpq-fe.h>


///Class with functions for DatabaseConnect and parser5.cpp
class DatabaseConnect{
private:
	PGconn *conn;
	std::string host = "129.24.26.137";
	std::string password = "Cerculsihr4T";
	std::string databasename = "postgres";
	std::string username = "postgres";

public:
	int getNextKey(std::string);///<Gets next available key from database
	int deleteTableContent(std::string);///<Deletes all table content from table specified by the user
	int makeTable(std::string);///<Creates a table with name specified by user
	int writeData(std::string, std::string, std::vector<std::string>);///<Uploads a row of data to table specified by user
	int connect();///<Connectes to database using credientials specified by DatabaseConnect(std::string, std::string, std::string, std::string)
	int disconnect();///<Disconnects user from database
	DatabaseConnect(std::string, std::string, std::string, std::string);///<User specifies credentials for connecting to database: database name, host ip address, username, password
	DatabaseConnect();///<Initialzes DatabaseConnect class with default values(non-functioning)
};
