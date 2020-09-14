"""
===============================================
Objective: Building email classifier with spacy
Author: Niteesha.Balla
Blog: https://dataaspirant.com
Date: 2020-08-30
===============================================
"""

# Reading the image
import matplotlib.image as mpimg
image = mpimg.imread('catimage.jpg')

# Reshape Image
import numpy as np
image = np.expand_dims(image, axis=0)

# ImageDataGenerator Instance creation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()

# View image
img = datagen.flow(image, batch_size=1)

from matplotlib import pyplot
image = next(img)[0].astype('uint8')
pyplot.imshow(image)
pyplot.axis('off')


# Rotation technique
datagen = ImageDataGenerator(rotation_range=50)


# Horizontal flip
datagen = ImageDataGenerator(horizontal_flip=True)

# Vertical flip
datagen = ImageDataGenerator(vertical_flip=True)

# Zoom
datagen = ImageDataGenerator(zoom_range=0.5)

# Width shift
datagen = ImageDataGenerator(width_shift_range=0.2)

# Height shift
datagen = ImageDataGenerator(height_shift_range=0.2)

# Brightness
datagen = ImageDataGenerator(brightness_range=[0.5, 1.5])
