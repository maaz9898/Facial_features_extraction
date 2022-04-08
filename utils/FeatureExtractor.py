import cv2
import mediapipe
import numpy as np
import skimage.exposure
from models.parser import face_parser
from keras import backend as K

"A class that encapsulates all functionalities needed"
class FeatureExtractor:
    # The constructor only needs the input image
    def __init__(self, image):
        self.img = image
        self.__results = []  # Mediapipe output
        self.__features = {}  # A dictionary to store features

    # Compress input image to a certain percentage (Default is 30%)
    def __compressImg(self, percentage=0.3):
        newWidth = int(self.img.shape[1] * percentage)
        newHeight = int(self.img.shape[0] * percentage)
        newDim = (newWidth, newHeight)  # Width and height after compression

        compressedImg = cv2.resize(self.img, newDim, interpolation = cv2.INTER_AREA)  # Resize to the new dimensions
        return compressedImg

    # Calls the compressImg() method and flips Red and Blue channels
    def __processImg(self, compress):
        if compress:
            compressedImg = self.__compressImg()
            copiedImg = compressedImg.copy()  
        else:
            copiedImg = self.img.copy()

        formattedImg = cv2.cvtColor(copiedImg, cv2.COLOR_BGR2RGB)

        face = mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=6, min_detection_confidence=0.5)

        self.__results = face.process(formattedImg)  # Run the FaceMesh module from Mediapipe on the preprocessed image

        return formattedImg.shape

    # Helper function
    def __euclideanDistance(self, leftx, lefty, rightx, righty):
        return np.sqrt((leftx-rightx)**2 +(lefty-righty)**2)

    # Helper funtion
    def __calcDistance(self, leftPt, rightPt, width, height):
        return int(
        self.__euclideanDistance(int(leftPt.x * width),  int(leftPt.y * height),
                           int(rightPt.x * width), int(rightPt.y * height)))
    
    # Helper function
    def __midpoint(self, pt1 , pt2, width, height):
        return int((pt1.x*width + pt2.x*width)/2), int((pt1.y*height + pt2.y*height)/2)


    # Main method that extracts different features
    def extractFeatures(self, faceOrient=True, extractMouth=True, compress=True):
        """ 
        A quick guide to different features IDs

        Left Eye Left Point --> 130
        Left Eye Right Point --> 243
        Left Eye Upper Point --> 27
        Left Eye Lower Point --> 23

        Right Eye Left Point --> 463
        Right Eye Right Point --> 359
        Right Eye Upper Point --> 257
        Right Eye Lower Point --> 253

        Face Center Point --> 5
        Face Left Point --> 93
        Face Right Point --> 323

        Mouth Left Point --> 61
        Mouth Right Point --> 291
        Upper Lip Upper Point --> 0
        Upper Lip Lower Point --> 13
        Lower Lip Upper Point --> 14
        Lower Lip Lower Point --> 17

        Hat Center --> 10
        """

        # Call the preprocessImg() method
        height, width, _ = self.__processImg(compress)

        # Proceed if face landmarks are detected
        if self.__results.multi_face_landmarks != None:
            counter = 0  # Keeps track of number of faces
            
            for facial_landmarks in self.__results.multi_face_landmarks:
                landmarks = facial_landmarks.landmark

                """ Left Eye Attributes """
                leftEyeW = self.__calcDistance(landmarks[130], landmarks[243], width, height)  # Width
                leftEyeH = self.__calcDistance(landmarks[27], landmarks[23], width, height)  # Height
                leftEyeMid = self.__midpoint(landmarks[27], landmarks[23], width, height)  # Midpoint

                leftEyeDX = int((landmarks[130].x) * width) - int((landmarks[243].x) * width)
                leftEyeDY = int((landmarks[130].y) * height) - int((landmarks[243].y) * height)
                leftEyeAngle = int(round((np.degrees(np.arctan2(leftEyeDY, leftEyeDX)) - 180), 0))  # Rotation Angle

                """ Right Eye Attributes """
                rightEyeW = self.__calcDistance(landmarks[463], landmarks[359], width, height)  # Width
                rightEyeH = self.__calcDistance(landmarks[257], landmarks[253], width, height)  # Height
                rightEyeMid = self.__midpoint(landmarks[257], landmarks[253], width, height)  # Midpoint

                rightEyeDX = int((landmarks[463].x) * width) - int((landmarks[359].x) * width)
                rightEyeDY = int((landmarks[463].y) * height) - int((landmarks[359].y) * height)
                rightEyeAngle = int(round((np.degrees(np.arctan2(leftEyeDY, leftEyeDX)) - 180), 0))  # Rotation Angle

                """ Hat Attributes """
                hatCenterX = int((landmarks[10].x) * width)
                hatCenterY = int((landmarks[10].y) * height)
                hatCenter = [hatCenterX, hatCenterY]  # Center
                hatAngle = int(round((np.degrees(np.arctan2(hatCenterX, hatCenterY)) - 180), 0))  # Rotation Angle

                # Update the features dictionary
                self.__features['Left_eye_' + str(counter + 1)] = {'rotation': leftEyeAngle, 'width': leftEyeW, 'height': leftEyeH,
                                                        'center': leftEyeMid}
                self.__features['Right_eye_' + str(counter + 1)] = {'rotation': rightEyeAngle, 'width': rightEyeW, 'height': rightEyeH,
                                                        'center': rightEyeMid}
                self.__features['Hat_' + str(counter + 1)] = {'rotation': hatAngle, 'center': hatCenter}

                # Find the face orientation if requested by the user
                if faceOrient:
                    faceLeftPt = [int((landmarks[93].x) * width), int((landmarks[93].y) * height)]
                    faceRightPt = [int((landmarks[323].x) * width), int((landmarks[323].y) * height)]
                    faceCenterPt = [int((landmarks[5].x) * width), int((landmarks[5].y) * height)]
                    
                    left2cen_dis = int(self.__euclideanDistance(faceLeftPt[0], faceLeftPt[1], faceCenterPt[0], faceCenterPt[1]))  # Left to center margin
                    right2cen_dis = int(self.__euclideanDistance(faceRightPt[0], faceRightPt[1], faceCenterPt[0], faceCenterPt[1]))  # Right to center margin

                    diff = abs(left2cen_dis - right2cen_dis)  # Margins difference
                    if diff > 10:
                        if left2cen_dis < right2cen_dis:
                            pos = 'Left'
                        elif left2cen_dis > right2cen_dis:
                            pos = 'right'
                        else:
                            pos = 'center'
                    else:
                        pos = 'center'

                    # Update the features dictionary
                    self.__features['Face_ori_' + str(counter + 1)] = {'Face_orientation': pos}

                if extractMouth:
                    mouthW = self.__calcDistance(landmarks[61], landmarks[291], width, height)  # Width
                    mouthH = self.__calcDistance(landmarks[0], landmarks[17], width, height)  # Height
                    mouthCenter = self.__midpoint(landmarks[13], landmarks[14], width, height)  # Center

                    # Update the features dictionary
                    self.__features['Mouth_' + str(counter + 1)] = {'Center': mouthCenter, 'width': mouthW, 'height': mouthH}

                counter += 1
            
            return self.__features


    # Helper function
    def __resizeImage(self, im, max_size=768):
        if np.max(im.shape) > max_size:
            ratio = max_size / np.max(im.shape)
            return cv2.resize(im, (0,0), fx=ratio, fy=ratio)

        return im


    # Returns a list of all faces detected
    def __detectFace(self, images):
        try:
            faces = []
            prs = face_parser.FaceParser()

            for image in images:
                # Resize image
                image = self.__resizeImage(image)

                output_image = image.copy()

                segmentation_output = prs.parse_face(image)[0]

                binary_mask = np.ones(shape=[segmentation_output.shape[0],segmentation_output.shape[1]]) * 255
                binary_mask[segmentation_output==16] = 0
                binary_mask[segmentation_output==15] = 0
                binary_mask[segmentation_output==14] = 0
                binary_mask[segmentation_output==0] = 0

                # blur alpha channel
                binary_mask = cv2.GaussianBlur(binary_mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

                # stretch so that 255 -> 255 and 127.5 -> 0
                binary_mask = skimage.exposure.rescale_intensity(binary_mask, in_range=(127.5, 255), out_range=(0,255))

                output_image[binary_mask<127.5] = 255

                faces.append(np.dstack((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), binary_mask)))
                # greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            K.clear_session()    
            return faces
        except:
            K.clear_session() 

    # Segments all faces detected
    def segmentFace(self, image):
        # image = cv2.imread(input)[..., ::-1]
        
        # Get the height and width of the input image.
        try:
            image_height, image_width, _ = image.shape

            cropped_images = []

            mp_face_detection = mediapipe.solutions.face_detection
            mp_face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

            # Perform the face detection on the image.
            results = mp_face_detector.process(image)

            # Check if the face(s) in the image are found.
            if results.detections:

                # Iterate over the found faces.
                for face_no, face in enumerate(results.detections):
                    face_bbox = face.location_data.relative_bounding_box

                    # Retrieve the required bounding box coordinates and scale them according to the size of original input image.
                    x1 = int(face_bbox.xmin * image_width)
                    y1 = int(face_bbox.ymin * image_height)

                    bbox_width = int(face_bbox.width * image_width)
                    bbox_height = int(face_bbox.height * image_height)

                    x2 = x1 + bbox_width
                    y2 = y1 + bbox_height

                    # Add some padding
                    x1 = x1 - int(bbox_width / 2)
                    y1 = y1 - int((bbox_height / 2) * 1.3)
                    x2 = x2 + int(bbox_width / 2)
                    y2 = y2 + int((bbox_height / 2) * 1.3)

                    if x1 < 0:
                        x1 = 0

                    if y1 < 0:
                        y1 = 0

                    if x2 > image_width:
                        x2 = image_width

                    if y2 > image_height:
                        x2 = image_height

                    # Crop the detected face region.
                    face_crop = image[y1:y2, x1:x2].copy()
                    cropped_images.append(face_crop)
                
                croppedFaces = self.__detectFace(cropped_images)
            K.clear_session()
            return croppedFaces
        except:
            K.clear_session()    


    # Upscales the image using super resolution
    def upscale(self, scalar):
        scaling_dict = {'2': "models/EDSR_x2.pb", '4': "models/EDSR_x4.pb"}  # A dict to select the appropriate model

        sr = cv2.dnn_superres.DnnSuperResImpl_create()  # SR opencv object

        path = scaling_dict[scalar]

        # Read and set the selected model
        sr.readModel(path)
        sr.setModel("edsr", int(scalar))

        # Upscales the input image
        copiedImg = self.img.copy()
        result = sr.upsample(copiedImg)

        return result