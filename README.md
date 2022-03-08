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
This project contains two main scripts
#### 1. features.py  
This script includes different features' extraction. The script takes command line arguments of selected features, the arguments include:  

* --input --> the path to an input image or directory (mandatory)  
* --face_orient --> set it to True if you want to find and save the face orientation to the output json file  
* --extract_mouth --> set it to True if you want to find and save the mouth's center, width and height to the output json file  
* --segment_face --> set it to True if you want to segment the faces inside the image  
* --extract_skintone --> set it to True if you want to extract the skin tone
  
Example command line argument:  
```
python features.py --input Test_data/ --segment_face True
```
  

#### 2. filter.py  
This script can apply different filters. The script takes command line arguments of selected features, the arguments include:  

* --input --> the path to an input image (mandatory)  
* --filter --> specifies the type of filter to use, the filters are:  
&nbsp;&nbsp;**'grey'** for Greyscale filter  
&nbsp;&nbsp;**'bright'** for Brightness Adjustment  
&nbsp;&nbsp;**'sharpen'** for Sharpening filter  
&nbsp;&nbsp;**'sepia'** for Sepia filter  
&nbsp;&nbsp;**'pencil_sketch'** for Pencil Sketch filter  
&nbsp;&nbsp;**'hdr'** for HDR filter  
&nbsp;&nbsp;**'summer'** for Summer filter  
&nbsp;&nbsp;**'winter'** for Winter filter  
  
  
Example command line argument:  
```
python filter.py --input Test_data/Test1.jpeg --filter pencil_sketch
```  
The output is saved inside the Filters directory.
