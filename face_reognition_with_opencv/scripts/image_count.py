# -*- coding: utf-8 -*-
"""
===============================================
Objective: Face Recognition With OpenCV
Author: Jaiganesh Nagidi
Blog: https://dataaspirant.com
Date: 2020-10-22
===============================================
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## load model
model = load_model("model/gender_predictor.model")

## definig classes
classes = ['men','women']

## pass input image
image = cv2.imread("images/lion.jpg")

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(gray_image,1.1,5)

## maintaining seperate counters
males=0
females=0

## iterating over faces

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),3)

    cropped_image = np.copy(image[y:y+h,x:x+w])

    ## preprocess the image according to our model
    res_face = cv2.resize(cropped_image, (96,96))
    #cv2.imshow("cropped image",res_face)
    res_face = res_face.astype("float") / 255.0
    res_face = img_to_array(res_face)
    res_face = np.expand_dims(res_face, axis=0)

    ## model prediction
    result = model.predict(res_face)[0]

    ## get label with max accuracy
    idx = np.argmax(result)
    label = classes[idx]
    print("predicted label is =",label)

    ## calculating count
    if label=="women":
        females=females+1
    else:
        males=males+1

cv2.rectangle(image,(0,0),(300,30),(255,255,255),-1)
cv2.putText(image," females = {},males = {} ".format(females,males),(0,15),
cv2.FONT_HERSHEY_TRIPLEX,0.6,(255, 101, 125),1)
cv2.putText(image," faces detected = " + str(len(faces)),(10,30),
cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1)


cv2.imshow("Gender FaceCounter",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
