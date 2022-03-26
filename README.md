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

## Deployed API  
The api.py script deploys the project to be usable by just passing arguments to an API.  

#### How it works:  
Run the api.py script by running  
```
python api.py
```  
This will initialize the app in the localhost and provide a URL where the app is running  

There are 3 endpoints: /features, /filters and /image.  

#### /features endpoint:  
* The features endpoint takes two arguments, the path to an input image and the scale parameter (optional)  
* The user can scale the input image by x2 or x4 by providng scale=2 or scale=4, if no scale parameter is provided, the input image is saved as it as  
* Images are saved inside static/output/ directory with a prefix that describes the operation performed on the image  
* Example  
```
http://127.0.0.1:5000/image?img=Test1.jpg&scale=2
```  
Note: the input image path is provided relative to the parent directory. In the previous example the Test1.jpg image is inside the parent directory.  
* This returns a json with the extracted features along with a URL of the masked photo

#### /filters endpoint:  
* The filters endpoint takes two arguments, the path to an input image and the filter type 
* Example  
```
http://127.0.0.1:5000/filters?image=static/output/upscaled_Test1.jpg&filter=grey
```  
* This endpoint also returns a json with a URL of the output image  


#### /image endpoint:  
* The image endpoint makes images accessible through the API and also displays images that are passed to it
* Example  
```
http://127.0.0.1:5000/image?img=../static/output/grey_Test1.jpg
```   
Note: input path is relative to the templates/ folder since this is where the html file is  

<img src="https://i.ibb.co/dMV8nYy/Screenshot-from-2022-03-26-19-53-14.png" alt="Screenshot-from-2022-03-26-19-53-14" border="0"></a>
