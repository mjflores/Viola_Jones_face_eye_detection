# OpenCv-face-detection

Este programa es una modificación del código presentado en "Implementing Face Detection using Python and OpenCV"
https://medium.com/analytics-vidhya/how-to-build-a-face-detection-model-in-python-8dc9cecadfe9

#
import cv2
import numpy as np
import glob as gl

facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

txtfiles = [] 
for file in gl.glob("*.jpeg"):
    txtfiles.append(file)
    print(file)

for ix in txtfiles:
    img = cv2.imread(ix,cv2.IMREAD_COLOR)
    imgColor = img.copy()
    imgTest = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    faces   = facecascade.detectMultiScale(imgTest, scaleFactor=1.2, minNeighbors=5)
    print('Total number of Faces found',len(faces))
#    
    for (x, y, w, h) in faces:
        cv2.rectangle(imgColor, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray  = imgTest[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(imgColor,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(255,0,255),2)
#
    cv2.imshow('Imagen',imgColor)
    if cv2.waitKey(1000) & 0xFF == 27:
        break

