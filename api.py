from flask import Flask, render_template, request
from flask_restful import Api
from utils.FeatureExtractor import FeatureExtractor
from filter import applyFilter
import numpy as np
import cv2
import os
from config import WRITE_PATH, IMG_PATH, FEATURES_PATH, PORT_NUM, LOG_FILE
import logging
from utils.helper import read_image_from_url, extract_image_name

def create_app():
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, force=True, encoding = 'utf-8')
    print("Log stored at: " + logging.getLoggerClass().root.handlers[0].baseFilename)
    #create static/output dir if not already exists
    if not os.path.exists(WRITE_PATH):
        logging.info('Creating output folder')
        os.makedirs(WRITE_PATH)
    logging.info('Starting Flask Server')
    app = Flask(__name__, static_url_path = "/static", static_folder = "static")
    return app

# Create a Flask app
app = create_app()

# Create an API using Flask app
api = Api(app)

@app.route('/mask/features')
def get_features():
    try:
        imageUrl = request.args.get('image')
        scale = request.args.get('scale')

        image = read_image_from_url(imageUrl)
        iimage = extract_image_name(imageUrl)

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
        out_path += iimage
        return_path += iimage
            
        features = FExtractor.extractFeatures()  # Call the extractFeatures() method
        cv2.imwrite(out_path, image)
        # Return a json with the extracted features + masked photo's URL
        data = {
            'photo': return_path,
            'data': features}
        return data, 200  # return data with 200 OK
    except Exception as e:
        logging.exception(f"Exception in /mask/features: {imageUrl}\n{e}")
        return "Error"


@app.route('/mask/segment')
def segment():
    try:
        imageUrl = request.args.get('image')

        image = read_image_from_url(imageUrl)
        iimage = extract_image_name(imageUrl)
        
        FExtractor = FeatureExtractor(image)  
        
        faces = FExtractor.segmentFace(image)  # Create a FeatureExtractor object
        json_arr = []
        for i,face_image in enumerate(faces):
            fname = f"_{str(i)}_{iimage}"
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
        logging.exception(f"Exception in /mask/segment: {imageUrl}\n{e}")
        return "Error"


# Filters class to use for creating filters endpoint
@app.route('/mask/filter')
def get():
    logging.debug('Called Endpoint: /mask/filter')
    try:
        imageUrl = request.args.get('image')
        fiilter = request.args.get('filter')
        
        image = read_image_from_url(imageUrl)
        iimage = extract_image_name(imageUrl)
        
        # Apply selected filter
        filteredImg = applyFilter(image, fiilter)

        # Create write path and URL
        out_path = WRITE_PATH + fiilter + '_' + iimage
        return_path = IMG_PATH + fiilter + '_' + iimage

        cv2.imwrite(out_path, filteredImg)

        # Return a json with the filtered photo's URL
        data = {
            'filtered_photo': return_path}
        return data, 200  # return data with 200 OK
    except Exception as e:
        logging.exception(f"Exception in /mask/filter: {imageUrl}\n{e}")
        return "Error"


# Image endpoint
@app.route('/mask/image')
def display_img():
    logging.debug('Called Endpoint: /mask/image')
    try:
        imageUrl = request.args.get('image')
        # Return the html displaying the input image
        return render_template("index.html", user_image = imageUrl)
    except Exception as e:
        logging.exception(f"Exception in /mask/image: {imageUrl}\n{e}")
        return "Error"

# Log Endpoint
@app.route('/mask/log')
def view_log():
    logging.debug('Called Endpoint: /mask/log')
    try:
        # read the log file
        log = open(LOG_FILE, 'r')
        log_data = log.read()
        return render_template("log.html", text = log_data)
    
    except Exception as e:
        logging.exception(f"Exception in /mask/log: {e}")
        return "Error"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT_NUM, debug=True)  # run Flask app
    
