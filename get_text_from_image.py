# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:33 2018

@author: SStarsman
"""

import cv2
import numpy as np
from keras.models import load_model

margin = 0
border = 2
char_height = 25
char_width = 14
white =[255, 255, 255]
tol = 3

# Library of symbols that can be translated
symbols =[]
for k in range(48,58):
    symbols.append(chr(k))
for k in range(65,91):
    symbols.append(chr(k))
for k in range(97, 123):
    symbols.append(chr(k))

try:
    temp = model.name
except:
    model = load_model('TextRecognition3C_256_62.h5')

im = cv2.imread('11-31-41-1602-(1~1498P) 434.jpg',0)
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
_,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #bound the images
    cv2.rectangle(im,(x,y),(x+w,y+h),(192,192,0),1)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = .5
fontColor              = (0,0,128)
lineType               = 1
indices = []
try:
    del images
except:
    print()
    
for idx, cnt in enumerate(contours):
#for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #following if statement is to ignore the noises and save the images which are of normal size(character)
    #In order to write more general code, than specifying the dimensions as 100,
    # number of characters should be divided by word dimension
    if w<(char_width * 1.7):
        if h<(char_height + tol) and h>(char_height - tol):         #save individual images
            sub_image = thresh1[y-margin:y+h+margin,x-margin:x+w+margin]
            new_sub_image = []
            if w > h:
                height = int((64. - 2 * border) / w * h)
                padding = 64 - height
                top_pad = int(padding / 2)
                bottom_pad = 64 - height - top_pad
                new_sub_image = cv2.resize(sub_image, dsize=(64, height))
                new_new_sub_image = cv2.copyMakeBorder( new_sub_image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=white)
            else:
                width = int((64. - 2 * border) / h * w)
                padding = 64 - width
                left_pad = int(padding / 2)
                right_pad = 64 - width - left_pad
                new_sub_image = cv2.resize(sub_image, dsize=(width, 64))
                new_new_sub_image = cv2.copyMakeBorder( new_sub_image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=white)
            new_new_sub_image = new_new_sub_image.reshape((64, 64, 1))
            final_sub_image = np.concatenate((new_new_sub_image, new_new_sub_image, new_new_sub_image), axis=2)
            try:
                images = np.append(images, np.reshape(final_sub_image, (1, 64, 64, 3)), axis=0)
            except Exception as ex:      
                images = np.reshape(final_sub_image, (1, 64, 64, 3))
            cv2.imwrite(str(idx)+".jpg", new_new_sub_image)
            indices.append(idx)
            cv2.rectangle(im,(x,y),(x+w,y+h),(128,128,0),1)
    else: # wide box
        if h< (char_height + tol) and h > char_height - tol:
            num_chars = round(w / char_width)
            # TO DO - get component images and characters iterate through each
            cv2.imwrite(str(idx) +".jpg", thresh1[y:y+h, x:x+w])
            for k in range(0, num_chars):
                sub_image = thresh1[y-margin:y+h+margin,x-margin + k*char_width:(x+margin + (k+1)*char_width-1)]
                new_sub_image = []
                width = int((64. - 2 * border) / h * w / num_chars)
                padding = 64 - width
                left_pad = int(padding / 2)
                right_pad = 64 - width - left_pad
                new_sub_image = cv2.resize(sub_image, dsize=(width, 64))
                new_new_sub_image = cv2.copyMakeBorder( new_sub_image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=white)
                new_new_sub_image = new_new_sub_image.reshape((64, 64, 1))
                final_sub_image = np.concatenate((new_new_sub_image, new_new_sub_image, new_new_sub_image), axis=2)
        #final_sub_image = np.concatenate(final_sub_image, new_new_sub_image)
                try:
                    images = np.append(images, np.reshape(final_sub_image, (1, 64, 64, 3)), axis=0)
                except Exception as ex:      
                    images = np.reshape(final_sub_image, (1, 64, 64, 3))
                cv2.imwrite(str(idx) + '_' + str(k) +".jpg", new_new_sub_image)
                indices.append(idx)
            cv2.rectangle(im,(x,y),(x+w,y+h),(128,128,0),1)

test = model.predict(images)
results = [symbols[i] for i in np.argmax(test, 1)]

x1 = 0
for idx, cnt_num in enumerate(indices):
    x,y,w,h = cv2.boundingRect(contours[cnt_num])
    if w < (char_width * 1.7):
        cv2.putText(im,results[idx], (x, y-5), font, fontScale, fontColor, lineType)
        x1 = 0
    else:
        cv2.putText(im,results[idx], (x + x1, y-5), font, fontScale, fontColor, lineType)
        x1 += char_width
cv2.imwrite("results.jpg", im)
cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
cv2.imshow('BindingBox',im)
cv2.waitKey(0)
#cv2.destroyWindow('BindingBox')