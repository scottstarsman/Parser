# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:29:11 2018

@author: SStarsman
"""
import random
import os

root_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/training/'
target_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/validation/'
filename_base = '/img001-0'
num_files = 1016
validation_split = 0.2
num_val_files = int(num_files * validation_split)

exception_list = []
cur_sample = 1
for sub_dir in os.listdir(root_dir):
    print(sub_dir)
    filename_base = sub_dir + '/img' + format(cur_sample,'03d') + '-'
    if not os.path.exists(target_dir + sub_dir):
        os.makedirs(target_dir + sub_dir)
    move_list = random.sample(range(num_files), num_val_files)
    for move_num in move_list:
        try:
            os.rename(root_dir + filename_base + format(move_num,'05d') + '.png', target_dir + filename_base + format(move_num,'05d') + '.png')
        except:
            exception_list.append(root_dir + filename_base + format(move_num,'05d') + '.png')
    cur_sample += 1