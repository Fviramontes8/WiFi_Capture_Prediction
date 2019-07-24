#WiFi_Capture_Prediction

This is the code used to parse and process captured data on a Wi-Fi network.

DatabaseConnector.py serves as an interface to a PostgreSQL database. It can create tables, read tables, write tables, read table names, delete tables, and checks if tables exist.

To view documentation you need doxygen, to download in linux:
	sudo apt install doxygen
or
	sudo apt-get install doxygen
To generate the documentation type the command:
	doxygen wifi_pred_config
It will generate two folders: html and latex. Inside you will find documentation on the code in this git.
