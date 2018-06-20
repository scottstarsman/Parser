# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:19:22 2018

@author: SStarsman
"""
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plot
import collections
from keras.models import load_model
from extract_text_from_image import extract_text
import pandas as pd

# Application settings
# Size of the equipment clusters
group_size = 275 
# Margins for the drawing contents. Clusters outside these margins are ignored.
margin_top = 455    
margin_left = 300
margin_bottom = 2490
margin_right = 4550
# Smallest number of contours in a cluster to be considered valid
min_cluster_size = 20
# margin around each extracted character
margin = 0
# white border placed around each character
border = 2
# The average tallness of characters (ratio of height to width)
ave_tallness = 1.7
char_height = 30
char_width = 14
tol = 6

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 3
fontColor              = (192, 0, 0)
lineType               = 4
blue = (192, 0, 0)
green = (0 ,192, 0)
red = (0, 0, 192)
cyan = (192, 192, 0)
light_cyan = (255, 255, 0)
white = (255, 255, 255)

def get_cluster(point, clusters):
    # Given a point in a 1X2 array and clusters in an NX2 array, find the cluster that is closest and return its index
    return np.argmin(np.linalg.norm(clusters-point, axis=1))

# Load the text recognition model
try: # Only load it if the model does not already exist.
    temp = model.name 
except:
    model = load_model('TextRecognition3C_256_62.h5')

im = cv2.imread('11-31-41-1602-(1~1498P) 434.jpg',0)
im_c = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
im_h, im_w = im.shape
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
_,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

characters, indices = extract_text(thresh1, contours, model, margin=margin, border=border, ave_tallness=ave_tallness, char_height=char_height, char_width=char_width, tol=tol, write_subimages=True)
X = np.array([])
Y = np.array([])

# Get additional information about each contour and perform additional processing
ratio = np.array([])
for idx, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    if x > margin_left and x+w < margin_right and y > margin_top and y+h < margin_bottom:
        # Calculate the ration of p^2 to A. The lowest real value is 4pi (circle)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter < 50: # find small contours and label them
            #cv2.putText(im_c, str(idx), (x, y), font, 0.75, (192, 0, 192), 1)
            #cv2.drawContours(im_c, [cnt], 0, (192, 0, 192), 1)
            if w < char_width*2 and w > char_width and h < int(char_height /4): # Find dashes
                characters.append('-')
                indices.append(idx)
                contours[idx] = contours[idx] - [0, int(char_height /2)] # offset the contour to the string baseline
        if area > 0.0:
            #print(str(perimeter**2 / area))
            ratio = np.append(ratio, perimeter**2 / area)
        else:
            ratio = np.append(ratio, float('inf'))
#        if area==0 or perimeter**2 / area > 60.: # Only line-like objects (long perimeter, small area)
#            if (w + h) > group_size: # Only longer lines
#                start_cluster = get_cluster(cnt[0, 0, :], cluster_centers)
#                end_cluster = get_cluster(cnt[-1, 0, :], cluster_centers)
#                start_cluster = get_cluster(np.array([x, y]), cluster_centers)
#                end_cluster = get_cluster(np.array([x+w, y+h]), cluster_centers)
                #if start_cluster != end_cluster:
                    #cv2.drawContours(im_c, [cnt], 0, green, 3)
                    #cv2.putText(im_c, str(idx), (x, y), font, 1, green, lineType)
            #print(str(area) + " " + str(perimeter))
    else:
        ratio = np.append(ratio, 0)

# Extract relevant contour data
first_element = True
for idx, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    if w < im_w/4 and h < im_h/4: # ignore contours that are too large
        if x > margin_left and x+w < margin_right and y > margin_top and y+h < margin_bottom:
            if first_element:
                X = np.array([x + w/2, y + h/2])
                X = np.reshape(X,(1,2))
                Y = np.array([0, y + h/2])
                Y = np.reshape(Y,(1,2))
                first_element = False
            else:
                X = np.append(X, np.reshape(np.array([x + w/2, y + h/2]), (1, 2)), 0)
                Y = np.append(Y, np.reshape(np.array([0, y + h/2]), (1, 2)), 0)

# Find clusters of the equipment
ms = MeanShift(bandwidth=group_size, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_sizes = collections.Counter(labels)
cluster_centers = ms.cluster_centers_
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)

# Find the dominant horizontal regions
bandwidth = estimate_bandwidth(Y) #, quantile=0.2, n_samples=500)
ms_y = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms_y.fit(Y)
labels_y = ms_y.labels_
cluster_sizes_y = collections.Counter(labels_y)
cluster_centers_y = ms_y.cluster_centers_

# Find text clusters. 
Z = np.array([])
first_element = True
x1 = 0
for idx, cnt_num in enumerate(indices):
    x,y,w,h = cv2.boundingRect(contours[cnt_num])
    if w < (char_width * 1.7):
        if first_element:
            Z = np.reshape(np.array([x, y]), (1, 2))
            first_element = False
        else:
            Z = np.append(Z, np.reshape(np.array([x, y]), (1, 2)), 0)
        #cv2.putText(im_c,characters[idx], (x, y-char_height), font, 1.0, cyan, lineType)
        x1 = 0
    else:
        if first_element:
            Z = np.reshape(np.array([x, y]), (1, 2))
            first_element = False
        else:
            Z = np.append(Z, np.reshape(np.array([x + x1, y]), (1, 2)), 0)        
        #cv2.putText(im_c,characters[idx], (x + x1, y-char_height), font, 1.0, cyan, lineType)
        x1 += char_width
text_x_factor = 0.1
Z[:, 0] = Z[:, 0] * text_x_factor
bandwidth_text = estimate_bandwidth(Z) #, quantile=0.2, n_samples=500)
bandwidth_text = 250 * text_x_factor
ms_text = MeanShift(bandwidth=bandwidth_text, bin_seeding=True)
ms_text.fit(Z)
labels_text = ms_text.labels_
cluster_centers_text = ms_text.cluster_centers_
labels_text_unique = np.unique(labels_text)

# Find most common hierarchy
u, u_cnt = np.unique(hierarchy[:,:,3],return_counts=True)
prime_area = u[np.argmax(u_cnt)]

# Convert characters into strings of related characters (words, tags, etc)
char_strings = []
char_string_positions = []
parent = []
is_boxed = []
for label in labels_text_unique:
    temp_chars = [characters[ind] for ind, x in enumerate(labels_text) if x==label]
    temp_contours = [contours[indices[ind]] for ind, x in enumerate(labels_text) if x==label]
    temp_indices = [indices[ind] for ind, x in enumerate(labels_text) if x==label]
    temp_hierarchy = [hierarchy[0, indices[ind], :] for ind, x in enumerate(labels_text) if x==label]
    
    left_pos = []
    for idx, character in enumerate(temp_chars):
        x,y,w,h = cv2.boundingRect(temp_contours[idx])
        if idx == 0 or temp_indices[idx] != temp_indices[idx-1]:
            left_pos.append(x)
            x1 = x
        else:
            x1 += char_width
            left_pos.append(x1)
    sort_indices = sorted(range(len(left_pos)), key=lambda k: left_pos[k])
    char_strings.append([])
    char_string_positions.append((x, y))
    parent.append(temp_hierarchy[0][3])
    for sidx, k in enumerate(sort_indices):
        x_p,y_p,w_p,h_p = cv2.boundingRect(contours[parent[-1]])
        if (h_p - h) / h < 0.2:
            is_boxed.append(True)
        else:
            is_boxed.append(False)
        if sidx>0:
            if left_pos[k] - left_pos[sort_indices[sidx-1]] > char_width * 2 or temp_hierarchy[k][3] != temp_hierarchy[sort_indices[sidx-1]][3]:
                # Start new string
                char_strings[-1] = ''.join(char_strings[-1])
                char_strings.append([])
                x,y,w,h = cv2.boundingRect(temp_contours[k])
                char_string_positions.append((x, y))
                parent.append(temp_hierarchy[k][3])

                #char_strings[-1].append(' ')
        char_strings[-1].append(temp_chars[k])
    char_strings[-1] = ''.join(char_strings[-1])

# Find all strings 
tags = [tag.upper() for tag in char_strings if tag.count('-') > 2]
        
# Find and draw connecting lines
minLineLength = 100
maxLineGap = 1
connectors = []
im_canny = cv2.Canny(im, 255/3, 255, 3)
lines = cv2.HoughLinesP(im_canny, rho=1, theta=np.pi/180, threshold=100, minLineLength=minLineLength, maxLineGap=maxLineGap)
for idx, line in enumerate(lines):
    if  line[0, 0] > margin_left and line[0, 2] < margin_right and line[0, 1] > margin_top and line[0, 3] < margin_bottom:
        start_cluster = get_cluster(line[:,0:2], cluster_centers)
        end_cluster = get_cluster(line[:,2:4], cluster_centers)
        if start_cluster != end_cluster:
            connectors.append(line.tolist())
            #cv2.line(im_c, (line[0, 0], line[0, 1]), (line[0,2], line[0,3]), green, 3)
            #cv2.putText(im_c, str(idx), (line[0,0], line[0,1]), font, 1, green, lineType)

# Prepare data for saving. Using a Pandas dataframe to export to Excel
data = pd.DataFrame(columns=['INSTRUMENT','PANEL','NODE','SLOT','I/O TYPE','LOCATION','MODEL','STATION NO','STRIP','TERMINAL NO', 'TERM_NUM','CABLE_NUM','CABLE_SET','WIRE_COLOUR','TERM_SIDE','CABLE_SIDE'])

            
# Annotate the image with cluster information
# Draw clusters
for idx, center in enumerate(cluster_centers):
    if cluster_sizes[idx] > min_cluster_size:
        cv2.circle(im_c, (int(center[0]), int(center[1])), group_size, blue, 2)
        text_size, baseline = cv2.getTextSize(str(cluster_sizes[idx]), font, fontScale, lineType)
        #cv2.putText(im_c,str(cluster_sizes[idx]), (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2)), font, fontScale, blue, lineType)

# Draw horizontal grouping
for idx, center in enumerate(cluster_centers_y):
    cv2.rectangle(im_c,(margin_left, int(center[1] - 2 * bandwidth)),(margin_right,int(center[1] + 2 * bandwidth)), red,5)

# Draw text clusters
for idx, center in enumerate(cluster_centers_text):
    #cv2.circle(im_c, (int(center[0] / text_x_factor), int(center[1])), int(bandwidth_text), cyan, 2)
    cv2.rectangle(im_c, (int((center[0] - bandwidth_text / 2) / text_x_factor), int(center[1])), (int((center[0] + bandwidth_text / 2) / text_x_factor), int(center[1] + bandwidth_text)), cyan, 2)
   
x1 = 0
for idx, cnt_num in enumerate(indices):
    x,y,w,h = cv2.boundingRect(contours[cnt_num])
    if w < (char_width * 1.7):
        #cv2.putText(im_c,characters[idx].upper(), (x, y-2), font, 1.0, cyan, 2)
        x1 = 0
    else:
        #cv2.putText(im_c,characters[idx].upper(), (x + x1, y-2), font, 1.0, cyan, 2)
        x1 += char_width
#cv2.drawContours(im_c, contours, -1, (0,255,0), 3)
for idx, char_list in enumerate(char_strings):
    if parent[idx] == prime_area:
        if char_list.count('-') > 2:
            temp_size = 1.2
        else:
            temp_size = 1.0
        temp_color = cyan
    else:
        temp_color = light_cyan
        temp_size = 1.0
    cv2.putText(im_c, char_list.upper(), char_string_positions[idx], font, temp_size, temp_color, 2)
    if is_boxed[idx]:
        cv2.rectangle(im_c, char_string_positions[idx], (char_string_positions[idx][0] + 20, char_string_positions[idx][1] + 20), red, 1)
        print(str(char_list) + " : " + str(char_string_positions[idx]))

# Draw connectors
for line in connectors:
    cv2.line(im_c, (line[0][0], line[0][1]), (line[0][2], line[0][3]), green, 3)
    
cv2.imwrite("cluster.jpg", im_c)
cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
cv2.imshow('BindingBox',im_c)
cv2.waitKey(0)