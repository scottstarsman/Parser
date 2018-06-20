# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:21:56 2018

@author: SStarsman
"""

from keras import models
from keras.models import load_model
from PIL import Image
import numpy as np

# Establish the library of symbols that the model can detect
symbols =[]
for k in range(48,58):
    symbols.append(chr(k))
for k in range(65,91):
    symbols.append(chr(k))
for k in range(97, 123):
    symbols.append(chr(k))
    
# Open the test image and scale it to 1X64X64X3 for the network
image = Image.open('324.jpg')
test_image = image.resize((64,64))
if image.mode != 'RGBA':
    #test_image = np.array(test_image)[:, :]
    test_image = np.resize(test_image, (64, 64, 1))
    layer = test_image
    test_image = np.append(layer, layer, axis=2)
    test_image = np.append(test_image, layer, axis=2)
else:
    test_image = np.array(test_image)[:, :, 0:3]
test_image = test_image.reshape(1, test_image.shape[1], test_image.shape[0], 3)
image.close()

# Only load the model if it doesn't exist
try:
    temp = model.name
except:
    model = load_model('TextRecognition3C_256_62.h5')

# Predict what the image is
test = model.predict(test_image)
result = symbols[np.argmax(test, 1)[0]]
print(result)