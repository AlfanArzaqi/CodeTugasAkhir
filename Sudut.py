# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:36:02 2023

@author: Alfan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:08:09 2023

@author: Alfan
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import time 
import pandas as pd
import math
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential, load_model
from google.protobuf.json_format import MessageToDict

def ExtrakHand (hands, image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    h,w,_ = image.shape
    black_image = np.zeros((h, w, 3), dtype = "uint8")
    
    handlandmark = []
    listimage = []
    mergeimage = []
    handsType = []
    kiriImage = []
    kananImage = []
    angular = []
    cropped = np.zeros((128,128,3), dtype="uint8")
    resized = Image.fromarray(cropped, 'RGB')

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      # for hand_handedness in results.multi_handedness :
      #   handType = MessageToDict(hand_handedness)['classification'][0]['label']
      for hand_landmarks in results.multi_hand_landmarks:
        for hand_handedness in results.multi_handedness :
          handType = MessageToDict(hand_handedness)['classification'][0]['label']
          hand_coor = np.zeros([21,3])
          for i in range(21):
               hand_coor[i,0]=hand_landmarks.landmark[i].x*w;
               hand_coor[i,1]=hand_landmarks.landmark[i].y*h;
               hand_coor[i,2]=hand_landmarks.landmark[i].z;
          
          handlandmark.append(hand_coor)
          
          minx_hand = min(hand_coor[:,0]) - 20
          miny_hand = min(hand_coor[:,1]) - 20
          
          maxx_hand = max(hand_coor[:,0]) + 20
          maxy_hand = max(hand_coor[:,1]) + 20
          
          mp_drawing.draw_landmarks(
              black_image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          
          # cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
          cv2.rectangle (black_image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
          
          if (int (minx_hand) < 0) :
              minx_hand = 0
          elif (int (miny_hand) < 0):
              miny_hand = 0
          elif (int (maxx_hand) < 0):
              maxx_hand = 0
          elif (int (maxy_hand) < 0):
              maxy_hand = 0
          
          cropped = black_image[int (miny_hand):int (maxy_hand), int (minx_hand):int (maxx_hand)]
          fliped = cv2.flip(cropped, 1)
          resized = cv2.resize(fliped,(128, 128))
          # listimage.append(resized)
          # mergeimage = np.hstack(listimage)
          
        
          kiriKoor = np.zeros([2])
          kananKoor = np.zeros([2])
          Startpoint = (0,0)
          Endpoint = (0,0)
          if handType == 'Right':
            cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
            cv2.putText(image, "Kiri", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            print("Kiri")
            kiriKoor[0]=hand_landmarks.landmark[i].x*w;
            kiriKoor[1]=hand_landmarks.landmark[i].y*h;
            Startpoint = (kiriKoor[0], kiriKoor[1])
            
          elif handType == 'Left':
            cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,255,0), 2)
            cv2.putText(image, "Kanan", (460, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            print("Kanan")
            kananKoor[0]=hand_landmarks.landmark[i].x*w;
            kananKoor[1]=hand_landmarks.landmark[i].y*h;
            Endpoint = (kananKoor[0], kananKoor[1])
            
          # cv2.line(image, Startpoint, Endpoint, (0, 255, 0), 3)
          
          X = kiriKoor[0] - kananKoor[0]
          Y = kiriKoor[1] - kananKoor[1]
          
          angular = math.atan((Y/X)*(180/math.pi))
          # print(angular)
          # angular = int (angular)

    return handlandmark, image, mergeimage, black_image, angular

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

NoKamera = 0
timeStart = time.time() 
timeNow = time.time()  
frameRate = 10
counter = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
      
    handlandmark, imagelandmark, mergeimage, black_image, angular = ExtrakHand(hands, image)  
    results_hand = hands.process(image)
    angular = str(angular)
    cv2.putText(imagelandmark, angular, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(imagelandmark, 1))
    cv2.imshow('MediaPipe', cv2.flip(black_image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()