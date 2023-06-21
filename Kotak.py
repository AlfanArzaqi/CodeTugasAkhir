import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import time 
import pandas as pd

def GetFileName():
        x = datetime.datetime.now()
        s = x.strftime('%Y-%m-%d-%H%M%S%f')
        return s
def CreateDir(path):
    ls = [];
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)   
    for i in range(len(ls)-2,-1,-1):
        print(ls[i])
        sf =ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

NoKamera = 0
timeStart = time.time() 
timeNow = time.time()  
frameRate = 10
counter = 0
maxCounter = 10
sNamaDirektori = GetFileName() 

sDirektoriData = "D:\\DATA\\SEMESTER 7\\PRATA\\Dataset\\"+sNamaDirektori 
CreateDir(sDirektoriData )

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(image)
    h,w,_ = image.shape
    black_image = np.zeros((h, w, 3), dtype = "uint8")
    koor = []
    koor2 = []
    hand_coor = np.zeros([21,3])
    y_all = []
    x_all = []
    x_max = 0
    y_max = 0
    x_min = 0
    y_min = 0

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            for _, landmark in enumerate(hand_landmarks.landmark):
              cx, cy = landmark.x * w, landmark.y * h
              y_all.append(cx) 
              x_all.append(cy) 
            
              x_max = max(y_all)
              y_max = max(x_all)
              x_min = min(y_all)
              y_min = min(x_all)
              
            # for i in range(21):
            #     hand_coor[i,0]=hand_landmarks.landmark[i].x*w;
            #     hand_coor[i,1]=hand_landmarks.landmark[i].y*h;
            #     hand_coor[i,2]=hand_landmarks.landmark[i].z*w;
            # koor.append(hand_coor)
            # koor2.append(hand_coor - hand_coor[0])
            # print(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z)
            
            # Semakin ke kiri nilai x semakin kecil, semakin ke bawah nilai y semakin besar, 
            # nilai z bernilai positif saat mendeteksi telapak tangan kanan dan akan bernilai 
            # negatif saat mendeteksi telapak tangan kiri dan akan menjadi lebih besar saat didekatkan ke kamera
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            mp_drawing.draw_landmarks(
                black_image,
                results_hand.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # minx_hand = min(hand_coor[:,0]) - 20
            # miny_hand = min(hand_coor[:,1]) - 20
            
            # maxx_hand = max(hand_coor[:,0]) + 20
            # maxy_hand = max(hand_coor[:,1]) + 20
            

        
            # if minx_hand > all(hand_coor[:,0]):
            #     minx_hand = hand_coor[:,0]
            # elif miny_hand > all(hand_coor[:,1]):
            #     miny_hand = hand_coor[:,1]
            # elif maxx_hand < all(hand_coor[:,0]):
            #     maxx_hand = hand_coor[:,0]
            # elif maxy_hand > all(hand_coor[:,1]):
            #     maxy_hand = hand_coor[:,1]

            # cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
            # cv2.rectangle (black_image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (255,255,255), 2)
                
            # print (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z)
            
        # if (int (minx_hand) < 0) :
        #     minx_hand = 0
        # elif (int (miny_hand) < 0):
        #     miny_hand = 0
        # elif (int (maxx_hand) < 0):
        #     maxx_hand = 0
        # elif (int (maxy_hand) < 0):
        #     maxy_hand = 0
        
        # cropped = black_image[int (miny_hand):int (maxy_hand), int (minx_hand):int (maxx_hand)]
        # fliped = cv2.flip(cropped, 1)
        # resized = cv2.resize(fliped,(128, 128))    
        
        timeNow = time.time()  
        if timeNow - timeStart > 1 / frameRate :
            counter = counter + 1
            
            # if (x_max < 0): x_max = 0
            # elif (x_min < 0): x_min = 0
            # elif (y_max < 0): y_max = 0
            # elif (y_min < 0): y_min = 0
            
            # cv2.rectangle (image, (int (x_min), int (y_min)), (int (x_max), int (y_max)), (0,0,0), 2)
            # cv2.rectangle (black_image, (int (x_min), int (y_min)), (int (x_max), int (y_max)), (255,255,255), 2)
            
            cropped = black_image[int(y_min):int(y_max), int(x_min):int(x_max)]
            # resized = cv2.resize(cropped,(128, 128))  
            
            #ndt= len(image)
            #sfc =sDirektoriData+"\\"+ str(Counter)
            #filename=sfc+".jpg"
            #cv2.imwrite(filename, image)
            
            ndt= len(koor)
            sfc =sDirektoriData+"\\"+ "testing" +str(counter)
            filename=sfc+".png"
            cv2.imwrite(filename, cropped)
            cv2.imwrite(sfc + ".jpg", image)
            timeStart = timeNow
            # for i in range (ndt):
            #     sf = sfc+"_"+str(i)
            #     np.savetxt(sf+".txt", koor[i], fmt='%.5f', delimiter=";")
            #     # np.savetxt(sf+"2.txt", koor2[i], fmt='%.5f', delimiter=";")
            #     df = pd.DataFrame(hand_coor)
            #     df.to_csv(sf+'.csv', sep=';', float_format='%.5f', index=False, header=False)
            
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    cv2.imshow('Black Image', cv2.flip(black_image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        #cap.release()
        #cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()