/***********************************************************************
 * @file DatabaseConnect.cpp
 * @author Seth Decker, Francisco Viramontes
 * 
 * @brief Contains functions to interact with PostgreSQL database
 * 
 * Description: This program is intended to  act as a front to 
 * comunicate with an sql database via c++ code. Inputs vary and
 * the user should see each function for their inputs and outputs.
 * It only contains functions.
 * 
 **********************************************************************/
#include "DatabaseConnect.hpp"
#include <string>
#include <vector>

//Initializes variables to connect to PostgreSQL database
/*
DatabaseConnect::DatabaseConnect()
{
	databasename = "postgres";
	username = "postgres";
	password = "Cerculsihr4T";
	host = "129.24.26.137";
	
}*/

///Declares variables to log on to the database
DatabaseConnect::DatabaseConnect(std::string _databasename, std::string _host, std::string _username, std::string _password)
{
	databasename = _databasename;
	username = _username;
	password = _password;
	host = _host;
	
}

///Uses the credentials initialzed by DatabaseConnect::DatabaseConnect() to login into the database and then to check the integrity of the connection
int DatabaseConnect::connect()
{
	std::string conninfo =  "dbname=" + databasename + " " + " host=" + host + " " + " user=" + username + " " + " password=" + password;
	
	conn = PQconnectdb(conninfo.c_str());
	
	//Checks to see that the backend connection was successfully made
	if(PQstatus(conn) == CONNECTION_OK) {
		std::cout << "Connected to database!" << std::endl;
	}
    else if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "Connection to database failed: %s",
                PQerrorMessage(conn));
    }
	return 0;
}

///Deletes table off of PostgreSQL database that is specified by table_name
int DatabaseConnect::deleteTableContent(std::string table_name) {
	std::string sql_query = "DELETE FROM " + table_name;
	PQexec(conn, sql_query.c_str());
	PQclear(PQexec(conn, sql_query.c_str()));
	return 0;
}


/**
 * @brief Given a string input this function will make a table under the name of the input string. 
 * 
 * It will have these columns (all integers):
 * 
 * Key,
 * Timestamp,
 * Number of Users,
 * Bits,
 * Number of Packets,
 * Average signal strength,
 * Average data rate,
 * Bits sent via 802.11b,
 * Bits sent via 802.11g,
 * Bits sent via 802.11n
 * 
 * Returns a print statement saying that the table was made.
 */
int DatabaseConnect::makeTable2GHz(std::string table_name) {
	std::string sql_query = "Create Table "+ table_name +"(Key int PRIMARY KEY, ts int, NoU int, bits int, pkt_num int, sigS int, dR int, phyb int, phyg int, phyn int)";
	PQexec(conn, sql_query.c_str());
	std::cout << "Made table: " << table_name << std::endl;
	//To avoid memory leakage
	PQclear(PQexec(conn, sql_query.c_str()));
	return 0;
}

/**
 * @brief Given a string input this function will make a table under the name of the input string. 
 * 
 * It will have these columns (all integers):
 * 
 * Key,
 * Timestamp,
 * Number of Users,
 * Bits,
 * Number of Packets,
 * Average signal strength,
 * Average data rate,
 * Bits sent via 802.11a,
 * Bits sent via 802.11n
 * 
 * Returns a print statement saying that the table was made.
 */
int DatabaseConnect::makeTable5GHz(std::string table_name) {
	std::string sql_query = "Create Table "+ table_name +"(Key int PRIMARY KEY, ts int, NoU int, bits int, pkt_num int, sigS int, dR int, phya int, phyn int)";
	PQexec(conn, sql_query.c_str());
	std::cout << "Made table: " << table_name << std::endl;
	//To avoid memory leakage
	PQclear(PQexec(conn, sql_query.c_str()));
	return 0;
}

/**Grabs the highest key from the database (if it is 
* empty, 0) then adds one so that we can add more data to the 
* database.
*/
int DatabaseConnect::getNextKey(std::string key) {
	std::string query = "select * from " + key;
	int next_key = PQntuples(PQexec(conn, query.c_str())) + 1;
	return next_key;
}

/**The input takes a string and a vector of 9 "int" elements 
 * (converted to strings) as an input and writes the data from the
 * vector to a database of name of the string given. If there is there
 * is any data in the database beforehand, it will add to the database
 * without overriding the data that was there before.
 * Example:
 * std::vector<std::string> data;
 * data = ["1525368933", "45", "5682987", "61", "-76", "12", "87654",
 *           "6842", "4814587"];
 * writeData2GHz("table_name", "5", data);
 */
int DatabaseConnect::writeData2GHz(std::string table_name, std::string key, std::vector<std::string> data)
{ 
	//String to write to database in sql syntax
	std::string sql_query = "INSERT INTO "+ table_name +" (Key, ts, NoU, bits, pkt_num, sigS, dR, phyb, phyg, phyn) VALUES('"+key+"', '"+data[0]+"', '"+data[1]+"', '"+data[2]+"', '"+data[3]+"', '"+data[4]+"', '"+data[5]+"', '"+data[6]+"', '"+data[7]+"', '"+data[8]+"')";
	
	//Executes command to write to database
	PQexec(conn, sql_query.c_str());
	std::cout << "Wrote to database" << std::endl;
	//To avoid memory leakage
	PQclear(PQexec(conn, sql_query.c_str()));
	return 0;
}

/**The input takes a string and a vector of 8 "int" elements 
 * (converted to strings) as an input and writes the data from the
 * vector to a database of name of the string given. If there is there
 * is any data in the database beforehand, it will add to the database
 * without overriding the data that was there before.
 * Example:
 * std::vector<std::string> data;
 * data = ["1525368933", "45", "5682987", "61", "-76", "12", "87654",
 *           "4814587"];
 * writeData5GHz("table_name", "5", data);
 */
int DatabaseConnect::writeData5GHz(std::string table_name, std::string key, std::vector<std::string> data)
{ 
	//String to write to database in sql syntax
	std::string sql_query = "INSERT INTO "+ table_name +" (Key, ts, NoU, bits, pkt_num, sigS, dR, phya, phyn) VALUES('"+key+"', '"+data[0]+"', '"+data[1]+"', '"+data[2]+"', '"+data[3]+"', '"+data[4]+"', '"+data[5]+"', '"+data[6]+"', '"+data[7]+"')";
	
	//Executes command to write to database
	PQexec(conn, sql_query.c_str());
	std::cout << "Wrote to database" << std::endl;
	//To avoid memory leakage
	PQclear(PQexec(conn, sql_query.c_str()));
	return 0;
}


///Disconnects from the PostgreSQL database and prints out 'Disconnected'
int DatabaseConnect::disconnect() {
	PQfinish(conn);
	std::cout << "Disconnected" << std::endl;
	
	return 0;
}
