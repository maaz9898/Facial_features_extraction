## Setting up the environment

1. Clone this repository from github.

2. Open the folder in VSCode, PyCharm, or any editor with support for the python language.
  Download the community edition of Pycharm for your operating system: [Link](https://www.jetbrains.com/pycharm/download/)
  In this blog, you can find instructions of [pycharm IDE](https://www.tutorialspoint.com/pycharm/pycharm_installation.htm).

3. For this project, install the python version mentioned in .python-version file (Currently 3.10.4).
  # Recommended: Install pyenv and virtualenv on your operating system. (Use choco or github). 
  
  # FOR WINDOWS: 'choco install pyenv-win'  https://github.com/pyenv-win/pyenv-win https://rkadezone.wordpress.com/2020/09/14/pyenv-win-virtualenv-windows/
  # FOR LINUX: https://github.com/pyenv/pyenv OR https://github.com/pyenv/pyenv-installer
  pyenv update
  pyenv install -l
  pyenv install 3.10.4
  pyenv rehash
  pyenv global 3.10.4

4. Install virtualenv for managing the virtual environment
  python -m pip install –U virtualenv
  cd PROJECT_DIR # Where PROJECT_DIR is the location of the cloned repository
  python -m virtualenv env
  source env/bin/activate  # FOR LINUX
  .\env\Scripts\activate # FOR WINDOWS

## Libraries and frameworks  
This project was developed using Python 3.10.4.
After installation of the IDE, Python, and Environment Manager open Command line and write a below line.
```bash
  python -m pip install -r requirements.txt
```
Now press ‘enter’ all requirements will be installed respectively to the virtual environment.


## How to run this project locally
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

## How to run this project as an API
#### 1. api.py
command line argument:  
```
python -m api.py
```
# For more information, refer to README_Site.md