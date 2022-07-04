from flask import Flask, render_template, request
from flask_restful import Api
from utils.FeatureExtractor import FeatureExtractor
from filter import applyFilter
import numpy as np
import cv2
from urllib.request import urlopen, Request
import os
from config import WRITE_PATH, IMG_PATH, FEATURES_PATH, PORT_NUM, LOG_FILE
import logging

# Create a Flask app
app = Flask(__name__, static_url_path = "/static", static_folder = "static")

# Create an API using Flask app
api = Api(app)

@app.route('/mask/features')
def get_features():
    try:
        iimage = request.args.get('image')
        scale = request.args.get('scale')

        req = Request(iimage, headers={'User-Agent': 'Mozilla/5.0'})
        requested_url = urlopen(req)
        
        image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)
        iimage = iimage.replace("?", "a")
        iimage = iimage.replace("=", "b")
        if ('png' not in iimage and'jpg' not in iimage):
            iimage=iimage+'.png'

        # image = cv2.imread(iimage)  # Read the input image
        FExtractor = FeatureExtractor(image)  # Create a FeatureExtractor object

        out_path = WRITE_PATH
        return_path = IMG_PATH
        # If a scalar is provided
        if scale != None:
            # Apply OpenCV SR upscaling & save output image
            image = FExtractor.upscale(scale)
            FExtractor.img = image
            out_path += 'upscaled_'
            return_path += 'upscaled_'
        
        # Create write path and URL
        out_path += iimage.split('/')[-1]
        return_path += iimage.split('/')[-1]
            
        features = FExtractor.extractFeatures()  # Call the extractFeatures() method
        cv2.imwrite(out_path, image)
        # Return a json with the extracted features + masked photo's URL
        data = {
            'photo': return_path,
            'data': features}
        return data, 200  # return data with 200 OK
    except Exception as e:
        logging.exception(e)
        return "Error"


@app.route('/mask/segment')
def segment():
    try:
        iimage = request.args.get('image')

        req = Request(iimage, headers={'User-Agent': 'Mozilla/5.0'})
        requested_url = urlopen(req)
        
        image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)

        iimage = iimage.replace("?", "a")
        iimage = iimage.replace("=", "b")
        if ('png' not in iimage and 'jpg' not in iimage):
                iimage=iimage+'.png'
        
        FExtractor = FeatureExtractor(image)  
        
        faces = FExtractor.segmentFace(image)  # Create a FeatureExtractor object
        json_arr = []
        for i,face_image in enumerate(faces):
            fname = f"_{str(i)}_{iimage.split('/')[-1]}"
            out_path = f"{WRITE_PATH}{fname}"
            return_path = f"{IMG_PATH}{fname}"
            cv2.imwrite(out_path, face_image)

            FExtractor.img = face_image.astype(np.uint8)
            features = FExtractor.extractFeatures()
            face_data= {'face': return_path, 'features': features}
            json_arr.append(face_data)
        data = {'data':json_arr}
        return data, 200
    except Exception as e:
        logging.exception(e)
        return "Error"


# Filters class to use for creating filters endpoint
@app.route('/mask/filter')
def get():
    logging.debug('Called Endpoint: /mask/filter')
    try:
        iimage = request.args.get('image')
        fiilter = request.args.get('filter')
        req = Request(iimage, headers={'User-Agent': 'Mozilla/5.0'})
        requested_url = urlopen(req)
        
        image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)

        iimage = iimage.replace("?", "a")
        iimage = iimage.replace("=", "b")
        if ('png' not in iimage and'jpg' not in iimage):
                iimage=iimage+'.png'
        
        
        # Apply selected filter
        filteredImg = applyFilter(image,fiilter)

        # Create write path and URL
        out_path = WRITE_PATH + fiilter + '_' + iimage.split('/')[-1]
        return_path = IMG_PATH + fiilter + '_' + iimage.split('/')[-1]

        cv2.imwrite(out_path, filteredImg)

        # Return a json with the filtered photo's URL
        data = {
            'filtered_photo': return_path}
        return data, 200  # return data with 200 OK
    except Exception as e:
        logging.exception(e)
        return "Error"


# Image endpoint
@app.route('/mask/image')
def display_img():
    logging.debug('Called Endpoint: /mask/image')
    try:
        img = request.args.get('img')
        # Return the html displaying the input image
        return render_template("index.html", user_image = img)
    except Exception as e:
        logging.exception(e)
        return "Error"

# Log Endpoint
@app.route('/mask/log')
def view_log():
    logging.debug('Called Endpoint: /mask/log')
    try:
        # read the log file
        log = open(LOG_FILE, 'r')
        log_data = log.read()
        return log_data
    except Exception as e:
        logging.exception(e)
        return "Error"

if __name__ == '__main__':
    #create static/output dir if not already exists
    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)    
    logging.info('Starting Server...')
    app.run(host='0.0.0.0', port=PORT_NUM, debug=False)  # run our Flask app
    logging.info('Server Started...')
