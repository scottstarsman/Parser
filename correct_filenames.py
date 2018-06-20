# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:55:21 2018

@author: SStarsman
"""

import os

root_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/training/'
target_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/validation/'
target_dir = root_dir

for sub_dir in os.listdir(target_dir):
    print(sub_dir)
    for filename in os.listdir(target_dir + sub_dir):
        if len(filename) == 15:
            newname = filename[0:7]+'0'+filename[7:]
            os.rename(target_dir + sub_dir + '/' + filename, target_dir + sub_dir + '/' + newname)
            print(target_dir + sub_dir + '/' + filename + ' --> ' + target_dir + sub_dir + '/' + newname)
