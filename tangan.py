
import copy
import cv2
import mediapipe as mp
import numpy as np
import math 
import os
import datetime

import time 
  
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
            
def ExtrakLandmark(hands,img):
    br,kl ,w= img.shape
    image =copy.copy(img) 
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    #image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lHandLandmark = []
    lm =[]
    if results.multi_hand_landmarks:
      
      for hand_landmarks in results.multi_hand_landmarks:
        
       
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        lRes = np.zeros([21,3])
        for i in range(21):
            lRes[i,0]=hand_landmarks.landmark[i].x*kl;
            lRes[i,1]=hand_landmarks.landmark[i].y*br;
            lRes[i,2]=hand_landmarks.landmark[i].z*kl;
            
        lHandLandmark.append(lRes)
        p0 = lRes[0:1,:]
        p1 = lRes[5:6,:]
        p2 = lRes[17:18,:]
        vx = p1 - p0
        vd = p2 - p0
        vx = vx/np.linalg.norm(vx);
        vz =np.cross(vx,vd)
        vz = vz/np.linalg.norm(vz);
        vy = np.cross(vz,vx)
        m = np.zeros([4,4])
        m[0:3,0]=vx
        m[0:3,1]=vy
        m[0:3,2]=vz
        m[0:3,3]=p0
        m[3,3]=1
        lm.append(m)
        
    return lHandLandmark,image,lm

        
        
        
def NormRow(v):
    r=np.sum(np.abs(v)**2,axis=-1)**(1./2)
    
    return r
    
    
        
        
    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

NoKamera = 0
TimeStart = time.time() 
TimeNow = time.time() 
FrameRate = 5
Counter = 0
sNamaDirektori = GetFileName() 

sDirektoriData = "c:\\temp\\Data\\Kiri\\"+sNamaDirektori 
CreateDir(sDirektoriData )
va =[]
va.append([0,1,2]) 
va.append([1,2,3])
va.append([2,3,4])

va.append([0,5,6])
va.append([5,6,7])
va.append([6,7,8])


va.append([0,9,10])
va.append([9,10,11])
va.append([10,11,12])



va.append([0,13,14])
va.append([13,14,15])

va.append([14,15,16])


va.append([0,17,18])
va.append([17,18,19])
va.append([18,19,20])
 


va.append([1,0,5])
va.append([5,0,9])
va.append([9,0,13])
va.append([13,0,17])
va = np.array(va, dtype=np.uint32)

hands = mp_hands.Hands( model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) 

# For webcam input:
cap = cv2.VideoCapture(NoKamera)



while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue

 
  ldt,imageLandMark,lm = ExtrakLandmark(hands,image)
  
  TimeNow = time.time() 
  if TimeNow -TimeStart >1/FrameRate:
      ndt= len(ldt)
      Counter =Counter+1
      sfc =sDirektoriData+"\\"+ str(Counter)
      filename=sfc+".jpg"
      cv2.imwrite(filename, imageLandMark)
      nd =np.array([ndt])
      np.savetxt(sfc+".num", nd, delimiter=',')
      TimeStart=TimeNow
      for i in range(ndt):
        # pp =ldt[i]
        # m = np.linalg.inv(lm[i])
        # pd =np.ones([21,4])
        # pd[:,0:3]=pp
        # p =np.matmul(m,pd.transpose()).transpose()
        sf = sfc+"_"+str(i)
        # v1 =ldt[i][va[:,0]]-ldt[i][va[:,1]]
        # v2 =ldt[i][va[:,2]]-ldt[i][va[:,1]]
        # v3 =np.cross(v1,v2)
        # s = NormRow(v3)
        # c = np.sum(v1*v2,axis=1)
        # sd = np.arctan2(s,c)
        np.savetxt(sf+".txt", ldt[i], delimiter=',') 
        # np.savetxt(sf+".inv", p[:,0:3], delimiter=',') 
        # p=p/p[5,0]
        # np.savetxt(sf+".invnorm", p[:,0:3], delimiter=',') 
        # np.savetxt(sf+".sd", sd, delimiter=',') 
        
        
  # Flip the image horizontally for a selfie-view display.
  cv2.imshow('MediaPipe Hands', cv2.flip(imageLandMark, 1))
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()
cv2.destroyAllWindows()