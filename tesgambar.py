# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:42:48 2023

@author: Alfan
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from keras import metrics
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import time 
import socket
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential, load_model

#Define Path
model_path = 'D:\DATA\SEMESTER 8\TUGASAKHIR\Model\modelTraining666.h5'
model_weights_path = 'D:\DATA\SEMESTER 8\TUGASAKHIR\Model\weightsTraining666.h5'
test_path = 'D:\DATA\SEMESTER 8\TUGASAKHIR\Dataset\Testing30\Diam'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

my_socket = socket.socket()
port = 8876
# ip = "192.168.230.51"
ip = "192.168.93.118"
my_socket.connect((ip, port))
msg = str(0)

x = datetime.datetime.now()
s = x.strftime('%H.%M.%S.%f')


def predict(file):
    x = load_img(file, target_size=(256,128))
    y = img_to_array(x)
    y = np.expand_dims(y, axis=0)
    array = model.predict(y)
    result = array[0]
    
    #print(result)
    answer = np.argmax(result)
    plt.imshow(x)
    plt.show()
    if answer == 0:
        # print("Predicted: Kanan Diam")
        # cv2.putText(black_image, 'Diam', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4,)
        msg = 'n'
        print(s)
        my_socket.send(msg.encode('utf_8'))
    elif answer == 1:
        # print("Predicted: Kanan Maju")
        # cv2.putText(black_image, 'Kanan', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        msg = 'r'
        my_socket.send(msg.encode('utf_8'))
    elif answer == 2:
        # print("Predicted: Kiri Diam")
        # cv2.putText(black_image, 'Kiri', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        msg = 'l'
        my_socket.send(msg.encode('utf_8'))
    elif answer == 3:
        # print("Predicted: Kiri Maju")
        # cv2.putText(black_image, 'Maju', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        msg = 'f'
        my_socket.send(msg.encode('utf_8'))
    elif answer == 4:
        # print("Predicted: Kiri Maju")
        # cv2.putText(black_image, 'Mundur', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        msg = 'b'
        my_socket.send(msg.encode('utf_8'))
    elif answer == 5:
        # print("Predicted: Kiri Maju")
        # cv2.putText(black_image, 'Tembak', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        msg = 's'
        my_socket.send(msg.encode('utf_8'))

    return answer

#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
    
        print(ret[0] + '/' + filename)
        result = predict(ret[0] + '/' + filename)
        print(" ")