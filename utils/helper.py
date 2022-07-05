from urllib.request import urlopen, Request
import cv2
import numpy as np


def read_image_from_url(url:str):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    requested_url = urlopen(req)
    image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, -1)
    return image

def extract_image_name(url:str):
    imageName = url.replace("?", "a").replace("=", "b")
    if ('png' not in imageName and 'jpg' not in imageName):
            imageName=imageName+'.png'
    return imageName.split('/')[-1]