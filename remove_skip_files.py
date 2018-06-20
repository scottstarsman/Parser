# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:09:29 2018

@author: SStarsman
"""

import os

skip = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33, 34, 35, 36, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 93, 94, 95, 96, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        149, 150, 151, 152, 189, 191, 213, 214, 215, 216, 221, 222, 223, 224, 229, 230, 231, 232, 261, 262, 263, 264, 293, 294, 295, 296, 297, 298, 299, 300, 385, 386, 287, 388, 401, 402, 403, 404, 433, 434, 435, 436, 453, 454, 455, 456, 461, 462, 463, 464, 478, 479, 480, 481, 482, 483,
        501, 502, 503, 504, 529, 530, 531, 532, 629, 630, 631, 632, 701, 702, 703, 704, 753, 754, 755, 756, 757, 758, 759, 760, 777, 778, 779, 780, 782,791, 792,  805, 806, 807, 808, 825, 826, 
        827, 828, 853, 854, 855, 856, 873, 874, 875, 876, 878, 879, 880, 897, 898, 899, 900, 929, 930, 931, 932, 933, 934, 935, 936, 941, 942, 943, 944, 993, 994, 995, 996, 997, 998, 999, 1000, 1005, 1006, 1007, 1008]
train_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/training/'
val_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/validation/'
train_dir = val_dir
other_dir = 'c:/onedrive/projects/coursera/deeplearning/parser/letters/'
validation_split = 0.2

exception_list = []
cur_sample = 1
for sub_dir in os.listdir(train_dir):
    print(sub_dir)
    filename_base = sub_dir + '/img' + format(cur_sample,'03d') + '-'
    if not os.path.exists(other_dir + sub_dir):
        os.makedirs(other_dir + sub_dir)
    for move_num in skip:
        try:
            os.rename(train_dir + filename_base + format(move_num,'05d') + '.png', other_dir + filename_base + format(move_num,'05d') + '.png')
        except:
            exception_list.append(train_dir + filename_base + format(move_num,'05d') + '.png')
    cur_sample += 1