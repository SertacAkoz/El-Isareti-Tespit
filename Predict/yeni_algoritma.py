from PIL import Image

import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle 

from utils import CvFpsCalc


def preProcess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img / 255 
    return img 
        
pickle_in = open("model_trained_hand.p","rb")
model = pickle.load(pickle_in) 

cap = cv.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 120)

while True:
    ret, image = cap.read()
    
    
    
    img = np.asarray(image)
    img = cv.resize(img, (320, 120)) # input shape
    
    cv.imshow("resulition", img)
    
    img = preProcess(img)
    
    img = img.reshape(1,120, 320,1) # 1 adet resim, (120, 320) boytunda, channel = 1
    
    classIndex = int(model.predict_classes(img))

    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)

    if probVal > 0.7:
        if classIndex == 0:
            cv.putText(image, "Palm"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        elif classIndex == 1:
            cv.putText(image, "L"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        elif classIndex == 2:
            cv.putText(image, "Fist"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        elif classIndex == 3:
            cv.putText(image, "Thumb"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        elif classIndex == 4:
            cv.putText(image, "Ok"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        elif classIndex == 5:
            cv.putText(image, "C"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        
    cv.imshow("Hand Detection", image)
    
    
    
    
    
    key = cv.waitKey(1)
        
    if key == 27:  # ESC
        break
    
cap.release()
cv.destroyAllWindows()
    
    
    
    
    
    
    
