from flask import Flask, render_template
from flask_restful import Resource, Api, reqparse
from utils.FeatureExtractor import FeatureExtractor
from filter import applyFilter
import cv2

# Create a Flask app
app = Flask(__name__)

# Output folder where images are saved
WRITE_PATH = 'static/output/'

# Image endpoint URL
IMG_PATH = 'http://127.0.0.1:5000/image?img=../static/output/'

# Create an API using Flask app
api = Api(app)

# Features class to use for creating features endpoint
class Features(Resource):
    # Define GET method behaviour
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('image', required=True)  # add args
        parser.add_argument('scale', required=False)
        
        args = parser.parse_args()  # parse arguments to dictionary
        
        image = cv2.imread(args['image'])  # Read the input image
        FExtractor = FeatureExtractor(image)  # Create a FeatureExtractor object
        
        features = FExtractor.extractFeatures()  # Call the extractFeatures() method

        # If a scalar is provided
        if args['scale'] != None:
            # Create write path and URL
            out_path = WRITE_PATH + 'upscaled_' + args['image'].split('/')[-1]
            return_path = IMG_PATH + 'upscaled_' + args['image'].split('/')[-1]

            # Apply OpenCV SR upscaling & save output image
            upscaled_img = FExtractor.upscale(args['scale'])
            cv2.imwrite(out_path, upscaled_img)
        
        else:
            # Create write path and URL
            out_path = WRITE_PATH + args['image'].split('/')[-1]
            return_path = IMG_PATH + args['image'].split('/')[-1]

            cv2.imwrite(out_path, image)
         
        # Return a json with the extracted features + masked photo's URL
        return {
            'masked_photo': return_path,
            'data': features} , 200  # return data with 200 OK


# Filters class to use for creating filters endpoint
class Filters(Resource):
    # Define GET method behaviour
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('image', required=True)  # add args
        parser.add_argument('filter', required=True)
        
        args = parser.parse_args()  # parse arguments to dictionary
        
        image = cv2.imread(args['image'])  # Read the input image
        
        # Apply selected filter
        filteredImg = applyFilter(image, args['filter'])

        # Create write path and URL
        out_path = WRITE_PATH + args['filter'] + '_' + args['image'].split('/')[-1]
        return_path = IMG_PATH + args['filter'] + '_' + args['image'].split('/')[-1]

        cv2.imwrite(out_path, filteredImg)

        # Return a json with the filtered photo's URL
        return {
            'filtered_photo': return_path} , 200  # return data with 200 OK


# Image endpoint
@app.route('/image')
def display_img():
    parser = reqparse.RequestParser()  # initialize
    
    parser.add_argument('img', required=True)  # add args
    
    args = parser.parse_args()  # parse arguments to dictionary

    # Return the html displaying the input image
    return render_template("index.html", user_image = args['img'])


# Add API endpoints
api.add_resource(Features, '/features')
api.add_resource(Filters, '/filters')

if __name__ == '__main__':
    app.run()  # run our Flask app

