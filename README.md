## Installation of the Package manager and IDE

Anaconda is the package manager for python. The distribution comes with the Python interpreter and various packages related to machine learning and data science.
Basically, the idea behind Anaconda is to make it easy for people interested in those fields to install all (or most) of the packages needed with a single installation. 

1. Install Anaconda (Python 3.7) on your operating system. You can either download anaconda from the official site and install it on your own or you can follow these anaconda installation tutorials below.

Install Anaconda on Windows: [Link](https://medium.com/@GalarnykMichael/install-python-anaconda-on-windows-2020-f8e188f9a63d)

Install Anaconda on Mac: [Link](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072)

Install Anaconda on Ubuntu (Linux): [Link](https://medium.com/@GalarnykMichael/install-python-on-ubuntu-anaconda-65623042cb5a)

2.  Download the community edition of Pycharm for your operating system: [Link](https://www.jetbrains.com/pycharm/download/)
In this blog, you can find instructions of [pycharm IDE](https://www.tutorialspoint.com/pycharm/pycharm_installation.htm).



## Libraries and frameworks

After installation of IDE and package manager.
Open Command line and write a below line.

```bash
  pip install -r requirements.txt
```
Now press ‘enter’ all requirements will be installed respectively.


## How to run this project
This project contains two files.
1. Test segmentation 2.ipynb
You can run this in google colab notebook its basic section base block base file which contains individual code execution processes.
Just go to the repository or go to the project folder and open this file and you will see the google colab link where you can execute the file.

2. face_segmentation.py
To run this you could put this complete code in google colab to save your time.
In Google Colab cloud
```bash
  !python Facial_features_extraction.py -input Test_data/Test1.jpg -face_ori True
```
Or open Facial_features_extraction.py  file in CLI as mentioned below.
In local machine
```bash
  python Facial_features_extraction.py -input Test_data/Test2.jpg -face_ori True
```
There is Test_data folder where testing images are available.

- First arguement takes image in '-input' argument takes the image directory.
- Second arguement takes True value for storing face Direction(orientation).
argument takes the True value in'-face_ori'. But it is optional arguement. 

Facial_features_extraction.py will generate json file like, facial_features.json
if input image has two faces or many facial features, so Facial_features_extraction.py will extract the faciall features and store into json file. 
facial_features.json

**(NEW)**  
By using the same -input command line argument, the user can now provide a directory path to perform feature extraction on all images inside it at once. 
```bash
  python Facial_features_extraction.py -input Test_data/ -face_ori True
```

