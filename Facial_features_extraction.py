import numpy as np
import json
import cv2
import mediapipe
from matplotlib import pyplot as plt
import argparse
drawingModule = mediapipe.solutions.drawing_utils
faceModule = mediapipe.solutions.face_mesh
face = faceModule.FaceMesh(static_image_mode=True, max_num_faces=6,
                                         min_detection_confidence=0.5)

def euclidean_distance(leftx,lefty, rightx, righty):
  return np.sqrt((leftx-rightx)**2 +(lefty-righty)**2)

def midpoint(p1x,p1y ,p2x,p2y):
  return int((p1x + p2x)/2), int((p1y + p2y)/2)

def save_features(results):
  data = {}

  if results.multi_face_landmarks != None:
    i = 0  # len(results.multi_face_landmarks)
    for facial_landmarks in results.multi_face_landmarks:
      # left eye
      width_l = int(
        euclidean_distance(int((facial_landmarks.landmark[130].x) * wid), int((facial_landmarks.landmark[130].y) * hei),
                           int((facial_landmarks.landmark[243].x) * wid),
                           int((facial_landmarks.landmark[243].y) * hei)))
      height_l = int(
        euclidean_distance(int((facial_landmarks.landmark[27].x) * wid), int((facial_landmarks.landmark[27].y) * hei),
                           int((facial_landmarks.landmark[23].x) * wid), int((facial_landmarks.landmark[23].y) * hei)))
      mid_of_left_eye = midpoint(int((facial_landmarks.landmark[27].x) * wid),
                                 int((facial_landmarks.landmark[27].y) * hei),
                                 int((facial_landmarks.landmark[23].x) * wid),
                                 int((facial_landmarks.landmark[23].y) * hei))
      dY = int((facial_landmarks.landmark[130].y) * wid) - int((facial_landmarks.landmark[243].y) * wid)
      dX = int((facial_landmarks.landmark[130].x) * wid) - int((facial_landmarks.landmark[243].x) * wid)
      anglel = int(round((np.degrees(np.arctan2(dY, dX)) - 180), 0))
      # print('left',anglel)
      # print(mid_of_left_eye)
      # print(int(width_l))
      # print(int(height_l))
      #print('Left eye features have been save to facial_features.json')
      # right eye
      width_r = int(
        euclidean_distance(int((facial_landmarks.landmark[463].x) * wid), int((facial_landmarks.landmark[463].y) * hei),
                           int((facial_landmarks.landmark[359].x) * wid),
                           int((facial_landmarks.landmark[359].y) * hei)))
      height_r = int(
        euclidean_distance(int((facial_landmarks.landmark[257].x) * wid), int((facial_landmarks.landmark[257].y) * hei),
                           int((facial_landmarks.landmark[253].x) * wid),
                           int((facial_landmarks.landmark[253].y) * hei)))
      mid_of_right_eye = midpoint(int((facial_landmarks.landmark[257].x) * wid),
                                  int((facial_landmarks.landmark[257].y) * hei),
                                  int((facial_landmarks.landmark[253].x) * wid),
                                  int((facial_landmarks.landmark[253].y) * hei))
      dY = int((facial_landmarks.landmark[463].y) * wid) - int((facial_landmarks.landmark[359].y) * wid)
      dX = int((facial_landmarks.landmark[463].x) * wid) - int((facial_landmarks.landmark[359].x) * wid)
      angler = int(round((np.degrees(np.arctan2(dY, dX)) - 180), 0))
      # print('right',angler)
      # print(mid_of_right_eye)
      # print(int(width_r))
      # print(int(height_r))
      #print('Right eye features have been save to facial_features.json')

      hat_center = [int((facial_landmarks.landmark[10].x) * wid), int((facial_landmarks.landmark[10].y) * hei)]
      # print(hat_center)
      angleh = int(round((np.degrees(np.arctan2(int((facial_landmarks.landmark[10].x) * wid),
                                                int((facial_landmarks.landmark[10].y) * wid))) - 180), 0))
      # print(int(angleh))
      #print('Hat features have been save to facial_features.json')
      data['Left_eye' + str(i + 1)] = {'rotation': anglel, 'width': width_l, 'height': height_l,
                                       'center': mid_of_left_eye}
      data['Right_eye' + str(i + 1)] = {'rotation': angler, 'width': width_r, 'height': height_r,
                                        'center': mid_of_right_eye}

      data['Hat' + str(i + 1)] = {'rotation': angleh, 'center': hat_center}
      i += 1

  #print(data)
  out_file = open("facial_features.json", "w")

  json.dump(data, out_file, indent=6)

  out_file.close()

def compression(org):
  scale_percent = 30 # percent of original size\n",
  width = int(org.shape[1] * scale_percent / 100)
  height = int(org.shape[0] * scale_percent / 100)
  dim = (width, height)

  org =cv2.resize(org,dim, interpolation = cv2.INTER_AREA)
  return org





parser = argparse.ArgumentParser()

# Add arguments


parser.add_argument("-input",
	help="input_image")

# Indicate end of argument definitions and parse args
#args = parser.parse_args()
args = vars(parser.parse_args())
org = cv2.imread(args['input'])
# to reduce the size of an image
org  = compression(org)
# to crop the image of detected faces
image = org.copy()
imageb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face.process(imageb)
hei, wid, _ = image.shape
save_features(results)


#
# python Facial_features_extraction.py -input Test1.jpg