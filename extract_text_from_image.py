# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:33 2018

@author: SStarsman
"""

import cv2
import numpy as np
#from keras.models import load_model
#import pandas as pd

# Library of symbols that can be translated
symbols =[]
for k in range(48,58):
    symbols.append(chr(k))
for k in range(65,91):
    symbols.append(chr(k))
for k in range(97, 123):
    symbols.append(chr(k))

def extract_text(thresh1, contours, model, margin=0, border=2, ave_tallness=1.7, char_height=30, char_width=14, tol=6, write_subimages=False):
    indices = []
        
    for idx, cnt in enumerate(contours):
    #for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        tallness = h / w
        #following if statement is to ignore the noises and save the images which are of normal size(character)
        #In order to write more general code, than specifying the dimensions as 100,
        # number of characters should be divided by word dimension
        if tallness > 1.25 and tallness < 30.0:
            if h<(char_height + tol) and h>(char_height - tol):         #save individual images
                sub_image = thresh1[y-margin:y+h+margin,x-margin:x+w+margin]
                new_sub_image = []
                if w > h:
                    height = int((64. - 2 * border) / w * h)
                    padding = 64 - height
                    top_pad = int(padding / 2)
                    bottom_pad = 64 - height - top_pad
                    new_sub_image = cv2.resize(sub_image, dsize=(64, height))
                    new_new_sub_image = cv2.copyMakeBorder( new_sub_image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
                else:
                    width = int((64. - 2 * border) / h * w)
                    padding = 64 - width
                    left_pad = int(padding / 2)
                    right_pad = 64 - width - left_pad
                    new_sub_image = cv2.resize(sub_image, dsize=(width, 64))
                    new_new_sub_image = cv2.copyMakeBorder( new_sub_image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[255,255,255])
                new_new_sub_image = new_new_sub_image.reshape((64, 64, 1))
                final_sub_image = np.concatenate((new_new_sub_image, new_new_sub_image, new_new_sub_image), axis=2)
                try:
                    images = np.append(images, np.reshape(final_sub_image, (1, 64, 64, 3)), axis=0)
                except Exception as ex:      
                    images = np.reshape(final_sub_image, (1, 64, 64, 3))
                if write_subimages:
                    test = model.predict(np.reshape(images[-1], (1, 64, 64, 3)))
                    result = [symbols[i] for i in np.argmax(test, 1)]
                    print(result[0])
                    cv2.imwrite(str(idx) + '_' + result[0] + ".jpg", new_new_sub_image)
                indices.append(idx)
                #cv2.rectangle(im,(x,y),(x+w,y+h),(128,128,0),1)
        else: # wide box
            if h< (char_height + tol) and h > char_height - tol:
                num_chars = round(w / char_width)
                cur_char_width = round(w / num_chars)
                if write_subimages:
                    cv2.imwrite(str(idx) +".jpg", thresh1[y:y+h, x:x+w])
                for k in range(0, num_chars):
                    sub_image = thresh1[y-margin:y+h+margin,x-margin + k*cur_char_width:(x+margin + (k+1)*cur_char_width-1)]
                    new_sub_image = []
                    width = int((64. - 2 * border) / h * cur_char_width)
                    padding = 64 - width
                    left_pad = int(padding / 2)
                    right_pad = 64 - width - left_pad
                    new_sub_image = cv2.resize(sub_image, dsize=(width, 64))
                    new_new_sub_image = cv2.copyMakeBorder( new_sub_image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[255,255,255])
                    new_new_sub_image = new_new_sub_image.reshape((64, 64, 1))
                    final_sub_image = np.concatenate((new_new_sub_image, new_new_sub_image, new_new_sub_image), axis=2)
            #final_sub_image = np.concatenate(final_sub_image, new_new_sub_image)
                    try:
                        images = np.append(images, np.reshape(final_sub_image, (1, 64, 64, 3)), axis=0)
                    except Exception as ex:      
                        images = np.reshape(final_sub_image, (1, 64, 64, 3))
                    if write_subimages:
                        test = model.predict(np.reshape(images[-1], (1, 64, 64, 3)))
                        result = [symbols[i] for i in np.argmax(test, 1)]
                        cv2.imwrite(str(idx) + '_' + str(k) + '_' + result[0] +".jpg", new_new_sub_image)
                    indices.append(idx)
    
    test = model.predict(images)
    results = [symbols[i] for i in np.argmax(test, 1)]
    return results, indices


