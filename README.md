# WiFi Capture Prediction

This is the code used to parse and process captured data on a Wi-Fi network and format it so that it can be trained on a Gaussian Process (GP)

## File descriptions

DatabaseConnector.py serves as an interface to a PostgreSQL database. It can create tables, read tables, write tables, read table names, delete tables, and checks if tables exist.

DatabaseProcessor.py - TBD

SignalProcessor.py - TBD

## Library requirements (conda is recommended when available)

### psycopg2 for database capabilities

    pip install psycopg2

### numpy for numerical computations

     pip install numpy

or

    conda install numpy
 
### scipy's signal module for signal processing

    pip install scipy

or

    conda install scipy

### matplotlib for plotting figures

    pip install matplotlib

or

    conda install matplotlib

### scikit-learn for machine learning (now moving to PyTorch and GPyTorch)

    pip install scikit-learn

or
    
    conda install scikit-learn

### PyTorch (GPU version)

    pip install torch torchvision

or

    conda install pytorch torchvision cuda=x -c pytorch

Where `x` is CUDA version `9.2`, `10.1`, `10.2`.

### PyTorch (CPU version)

    pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

or

    conda install pytorch torchvision cpuonly -c pytorch

### GPyTorch

    pip install gpytorch

## Documenation generation

To view documentation you need doxygen, to download in linux:

    sudo apt install doxygen

or

    sudo apt-get install doxygen

To generate the documentation type the command: `doxygen wifi_pred_config`

It will generate two folders: html and latex. Inside you will find documentation on the code in this git.
