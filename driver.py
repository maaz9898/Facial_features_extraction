import argparse
import cv2
import json
import os
from SkinToneFinder import findSkinTone
from FeatureExtractor import FeatureExtractor


""" This is the driver script that uses the FeatureExtractor class"""
def main():
    # Define the argument parser along with the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image or directory")
    parser.add_argument("--face_orient", default=True, help="True for adding face orientation")
    parser.add_argument("--extract_mouth", default=True, help="True for adding mouth extraction")
    parser.add_argument("--segment_face", default=True, help="True for adding face segmentation")
    parser.add_argument("--extract_skintone", default=True, help="True for adding skin-tone extraction")
    args = vars(parser.parse_args())

    # Apply feature extraction on a whole directory
    if (len(args['input'].split('.jpg')) == 1) and (len(args['input'].split('.jpeg')) == 1) and (len(args['input'].split('.png')) == 1):
        for imgPath in os.listdir(args['input']):
            image = cv2.imread(args['input'] + imgPath)
            
            FExtractor = FeatureExtractor(image)  # Create a FeatureExtractor object

            features = FExtractor.extractFeatures(args['face_orient'], args['extract_mouth'])  # Call the extractFeatures() method

            writePath = args['input'] + imgPath.split('.')[0]  # Specify the path to write to
            
            out_file = open(writePath + 'json', "w")  # Create .json file with writing privilege

            # Write from the features dictionary then close the .json file
            json.dump(features, out_file, indent=4)
            out_file.close()

            if args['segment_face']:
                faces = FExtractor.segmentFace(imgPath)

                counter = 0
                # Iterate over the cropped faces images
                for face in faces:
                    cv2.imwrite(writePath + '_face_' + str(counter) + '.png', face)
                    counter +=1

            if args['extract_skintone']:
                findSkinTone(args['input'])
    
    # Apply feature extraction on a single image
    else:
        image = cv2.imread(args['input'])  # Read the input image
        FExtractor = FeatureExtractor(image)  # Create a FeatureExtractor object

        features = FExtractor.extractFeatures(args['face_orient'], args['extract_mouth'])  # Call the extractFeatures() method

        writePath = args['input'].split('.')[0]  # Specify the path to write .json to
        out_file = open(writePath + '.json', "w")  # Create .json file with writing privilege

        # Write from the features dictionary then close the .json file
        json.dump(features, out_file, indent=4)
        out_file.close()

        faces = FExtractor.segmentFace(args['input'])

        if args['segment_face']:
            counter = 0
            # Iterate over the cropped faces images
            for face in faces:
                # saving final transparent image
                cv2.imwrite(writePath + '_face_' + str(counter) + '.png', face)
                counter +=1

        if args['extract_skintone']:
            for face in faces:
                findSkinTone(args['input'])

if __name__ == '__main__':
    main()