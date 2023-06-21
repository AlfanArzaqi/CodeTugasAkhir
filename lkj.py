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
import socket
import pandas as pd
import threading
import queue
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential, load_model

#Define Path
model_path = 'D:\DATA\SEMESTER 8\TUGASAKHIR\Model\modelTraining6.h5'
model_weights_path = 'D:\DATA\SEMESTER 8\TUGASAKHIR\Model\weightsTraining6.h5'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

NoKamera = 0
timeStart = time.time() 
timeNow = time.time()  
frameRate = 10
counter = 0

prev_frame_time = 0
new_frame_time = 0

my_socket = socket.socket()
port = 8876
# ip = "192.168.230.51"
ip = "192.168.5.118"
my_socket.connect((ip, port))
msg = str(0)
msgrcv = str(0)

def ExtrakHand ():
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
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
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            h,w,_ = image.shape
            black_image = np.zeros((h, w, 3), dtype = "uint8")
            
            handlandmark = []
            listimage = []
            mergeimage = []
            cropped = np.zeros((128,128,3), dtype="uint8")
            resized = Image.fromarray(cropped, 'RGB')

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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
                    
                    cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
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
                    listimage.append(resized)
                    mergeimage = np.hstack(listimage)

                timeNow = time.time()  
                # fps = 1/(timeNow-timeStart)
                # fps = int(fps)
                # fps = str(fps)
                # cv2.putText(imagelandmark, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                # print(fps)
                # if timeNow - timeStart > 1 / frameRate :
                #     counter = counter + 1
                if len(handlandmark) == 2 :
                    
                    Im = Image.fromarray(mergeimage, 'RGB') 
                    Im = Im.resize((128, 256))
                    img_array = np.array(Im)
                    img_array = np.expand_dims(img_array, axis=0)
                    array = model.predict(img_array)
                    result = array[0]
                    answer = np.argmax(result)
                    print (answer)
                    if answer == 0:
                        # print("Predicted: Kanan Diam")
                        cv2.putText(black_image, 'Diam', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4,)
                        msg = 'n'
                        my_socket.send(msg.encode('utf_8'))
                    elif answer == 1:
                        # print("Predicted: Kanan Maju")
                        cv2.putText(black_image, 'Kanan', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                        msg = 'r'
                        my_socket.send(msg.encode('utf_8'))
                    elif answer == 2:
                        # print("Predicted: Kiri Diam")
                        cv2.putText(black_image, 'Kiri', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                        msg = 'l'
                        my_socket.send(msg.encode('utf_8'))
                    elif answer == 3:
                        # print("Predicted: Kiri Maju")
                        cv2.putText(black_image, 'Maju', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                        msg = 'f'
                        my_socket.send(msg.encode('utf_8'))
                    elif answer == 4:
                        # print("Predicted: Kiri Maju")
                        cv2.putText(black_image, 'Mundur', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                        msg = 'b'
                        my_socket.send(msg.encode('utf_8'))
                    elif answer == 5:
                        # print("Predicted: Kiri Maju")
                        cv2.putText(black_image, 'Tembak', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                        msg = 's'
                        my_socket.send(msg.encode('utf_8'))
                else :
                    cv2.putText(black_image, 'Tangan Tidak Terdeteksi', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                    msg = 'a'
                    my_socket.send(msg.encode('utf_8'))
                timeStart = timeNow
                      
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            cv2.imshow('MediaPipe', cv2.flip(black_image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                my_socket.shutdown(socket.SHUT_RDWR)
                my_socket.close()
                break
        cap.release()
        cv2.destroyAllWindows()
        
    return handlandmark, image, mergeimage, black_image

def RecShoot ():
    handlandmark, image, mergeimage, black_image = ExtrakHand()
    msgrcv2 = (my_socket.recv(1024).decode())
    print(msgrcv2,end='')
    if msgrcv == "FF":
        cv2.putText(black_image, 'Anda Tertembak', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        my_socket.shutdown(socket.SHUT_RDWR)
        my_socket.close()

if __name__ == "__main__":
    input_buffer = queue.Queue()
    detection_buffer = queue.Queue()

    t1 = threading.Thread(target=RecShoot, name='t1')
    t2 = threading.Thread(target=ExtrakHand, name='t2')
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# my_socket.shutdown()
# my_socket.close()