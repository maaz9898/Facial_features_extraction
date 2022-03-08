import cv2
import numpy as np
import scipy
import argparse
from scipy.interpolate import UnivariateSpline


#greyscale filter
def greyscale(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale


# brightness adjustment
def bright(img, beta_value):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright


#sharp effect
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen


#sepia effect
def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


#colour pencil sketch effect
def pencil_sketch_col(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_color


#HDR effect
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr


# Helper function
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))


#summer effect
def summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum


#winter effect
def winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win


# Apply selected filter
def applyFilter(img, filter_type, beta=60):
    if filter_type == 'grey':
        return greyscale(img) 

    elif filter_type == 'bright':
        return bright(img, beta)

    elif filter_type == 'sharpen':
        return sharpen(img)

    elif filter_type == 'sepia':
        return sepia(img)

    elif filter_type == 'pencil_sketch':
        return pencil_sketch_col(img)

    elif filter_type == 'hdr':
        return HDR(img)

    elif filter_type == 'summer':
        return summer(img)

    elif filter_type == 'winter':
        return winter(img)

    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image or directory")
    parser.add_argument("--filter", default=None)
    args = vars(parser.parse_args())

    image = cv2.imread(args['input'])
    filterType = str(args['filter'])

    out_path = 'Filters/' + args['input'].split('/')[1].split('.')[0] + '_'

    filteredImg = applyFilter(image, filterType)

    cv2.imwrite(out_path + filterType + '.png', filteredImg)

if __name__ == '__main__':
    main()