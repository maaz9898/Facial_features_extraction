from flask import Flask, render_template, request
from flask_restful import Api
from utils.FeatureExtractor import FeatureExtractor
from filter import applyFilter
import numpy as np
import cv2
from urllib.request import urlopen
from config import WRITE_PATH, IMG_PATH, FEATURES_PATH, PORT_NUM
# Create a Flask app
app = Flask(__name__, static_url_path = "/static", static_folder = "static")


# Create an API using Flask app
api = Api(app)

@app.route('/mask/features')
def get_features():
    try:
        iimage = request.args.get('image')
        scale = request.args.get('scale')

        requested_url = urlopen(iimage)
        
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
        # print(e)
        return "Error"


@app.route('/mask/segment')
def segment():
    try:
        iimage = request.args.get('image')

        requested_url = urlopen(iimage)
        
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
        print(e)
        return "Error"


# Filters class to use for creating filters endpoint
@app.route('/mask/filter')
def get():
    try:
        iimage = request.args.get('image')
        fiilter = request.args.get('filter')
        # parser = reqparse.RequestParser()  # initialize
        
        # parser.add_argument('image', required=True)  # add args
        # parser.add_argument('filter', required=True)
        
        # args = parser.parse_args()  # parse arguments to dictionary
        # nparr = np.fromstring(iimage, np.uint8)
        # image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        requested_url = urlopen(iimage)
        
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
        # print(e)
        return "Error"


# Image endpoint
@app.route('/mask/image')
def display_img():
    try:
        img = request.args.get('img')
        # parser = reqparse.RequestParser()  # initialize
        
        # parser.add_argument('img', required=True)  # add args
        
        # args = parser.parse_args()  # parse arguments to dictionary

        # Return the html displaying the input image
        return render_template("index.html", user_image = img)
    except Exception as e:
        # print(e)
        return "Error"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT_NUM, debug=False)  # run our Flask app
