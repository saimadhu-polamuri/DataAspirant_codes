"""
===============================================
Objective: Building email classifier with spacy
Author: Niteesha.Balla
Blog: https://dataaspirant.com
Date: 2020-08-30
===============================================
"""


#load the packages
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import matplotlib.image as mpimg
import numpy as np

#load the cat image
image = mpimg.imread('catimage.jpg')


#reshape the image into (batch_size, height, width, channels)
image = np.expand_dims(image, axis=0)

#view the shape of the image
image.shape

#apply transformations using ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=50,
                             horizontal_flip=True,
                             vertical_flip=True,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             brightness_range=[0.5, 1.5])

#display the transformations of the image
img = datagen.flow(image, batch_size=1)
fig, ax = pyplot.subplots(nrows=1, ncols=3, figsize=(15,15))
for i in range(3):
  image = next(img)[0].astype('uint8')
  ax[i].imshow(image)
  ax[i].axis('off')
