"""
===============================================
Objective: Face Recognition With OpenCV
Author: Jaiganesh Nagidi
Blog: https://dataaspirant.com
Date: 2020-10-22
===============================================
"""

## This only for blog post, please consider the other files.

## Importing necessary libraries
import numpy as np
import random
import cv2
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model


## initial parameters
lr = 1e-2
batch_size = 32
epochs = 100
img_dims = (96,96,3) #specifying image dimensions

data = []
labels = []

# loading image files
image_files = [f for f in glob.glob(r'C:\Desktop\gender_dataset' + "/**/*",
recursive=True) if not os.path.isdir(f)]

random.shuffle(image_files)

## converting images to arrays
for img in image_files:

    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    ## labelling the categories
    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])  # [[1], [0], [0], ...]


## pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

## split dataset for training and validation
x_train,x_test,y_train,y_test = train_test_split(data, labels,
test_size=0.2,random_state=42)

## converting into categorical labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# augmenting the dataset
aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")



## Defining the  Convolutional Model

## defining input shape
width = img_dims[0]
height = img_dims[1]
depth = img_dims[2]
inputShape = (height, width, depth)
dim = -1

# model creation

model = Sequential()

model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))

model.add(Conv2D(256, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation("sigmoid"))


## compile the model
opt = Adam(lr=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

## fit the model
h = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test,y_test),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs, verbose=1)

## save the model
model.save('gender_predictor.model')


## Load the gender predicition model

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

#load model
model = load_model("gender_predictor.model")

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#defining classes
classes = ['men','women']


def gender_facecounter(image, m, f, size=0.5):
    ## convert image into gray scaled image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray_image, 1.1,5)
    ## iterating over faces

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),3)

        cropped_image = np.copy(image[y:y+h,x:x+w])

        ## preprocess the image according to our model
        res_face = cv2.resize(cropped_image, (96,96))
        ## cv2.imshow("cropped image",res_face)
        res_face = res_face.astype("float") / 255.0
        res_face = img_to_array(res_face)
        res_face = np.expand_dims(res_face, axis=0)


        ## model prediction
        result = model.predict(res_face)[0]

        ## get label with max accuracy
        idx = np.argmax(result)
        label = classes[idx]

        ## calculating count
        if label == "women":
            f = f+1
        else:
            m = m+1

    cv2.rectangle(image,(0,0),(300,30),(255,255,255),-1)
    cv2.putText(image, " females = {},males = {} ".format(f,m),(0,15),
    cv2.FONT_HERSHEY_TRIPLEX,0.6,(255, 101, 125),1)
    cv2.putText(image, " faces detected = " + str(len(faces)),(10,30),
    cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1)

    return image


## loading an image
image = cv2.imread("pic3.jpg") #path to image

## maintaining separate counters
males = 0
females = 0

cv2.imshow("Gender FaceCounter", gender_facecounter(image,males,females ))
cv2.waitKey(0)
cv2.destroyAllWindows()


##  For video demo
source = cv2.VideoCapture(0)

while True:
    ret, frame = source.read()
    x = 0
    y = 0
    cv2.imshow("Live Facecount", gender_facecounter(frame, x, y))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

source.release()
cv2.destroyAllWindows()
