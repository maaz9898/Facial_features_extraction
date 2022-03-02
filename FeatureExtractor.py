import cv2
import mediapipe
import numpy as np

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
    def extractFeatures(self, faceOrient, extractMouth, compress=True):
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
                    self.__features['Mouth' + str(counter + 1)] = {'Center': mouthCenter, 'width': mouthW, 'height': mouthH}

                counter += 1
            
            return self.__features