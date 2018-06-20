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

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 3
fontColor              = (192, 0, 0)
lineType               = 4
blue = (192, 0, 0)
green = (0 ,192, 0)
red = (0, 0, 192)

def get_cluster(point, clusters):
    # Given a point in a 1X2 array and clusters in an NX2 array, find the cluster that is closest and return its index
    return np.argmin(np.linalg.norm(clusters-point, axis=1))
    
im = cv2.imread('11-31-41-1602-(1~1498P) 434.jpg',0)
im_c = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
im_h, im_w = im.shape
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
_,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
X = np.array([])
Y = np.array([])
    
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

# Find and draw connecting lines
ratio = np.array([])
for idx, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    if x > margin_left and x+w < margin_right and y > margin_top and y+h < margin_bottom:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if area > 0.0:
            #print(str(perimeter**2 / area))
            ratio = np.append(ratio, perimeter**2 / area)
        else:
            ratio = np.append(ratio, float('inf'))
        if area==0 or perimeter**2 / area > 60.: # Only line-like objects (long perimeter, small area)
            if (w + h) > group_size: # Only longer lines
                start_cluster = get_cluster(cnt[0, 0, :], cluster_centers)
                end_cluster = get_cluster(cnt[-1, 0, :], cluster_centers)
                start_cluster = get_cluster(np.array([x, y]), cluster_centers)
                end_cluster = get_cluster(np.array([x+w, y+h]), cluster_centers)
                #if start_cluster != end_cluster:
                    #cv2.drawContours(im_c, [cnt], 0, green, 3)
                    #cv2.putText(im_c, str(idx), (x, y), font, 1, green, lineType)
            #print(str(area) + " " + str(perimeter))
    else:
        ratio = np.append(ratio, 0)
minLineLength = 100
maxLineGap = 1
im_canny = cv2.Canny(im, 255/3, 255, 3)
lines = cv2.HoughLinesP(im_canny, rho=1, theta=np.pi/180, threshold=100, minLineLength=minLineLength, maxLineGap=maxLineGap)
for idx, line in enumerate(lines):
    if  line[0, 0] > margin_left and line[0, 2] < margin_right and line[0, 1] > margin_top and line[0, 3] < margin_bottom:
        start_cluster = get_cluster(line[:,0:2], cluster_centers)
        end_cluster = get_cluster(line[:,2:4], cluster_centers)
        if start_cluster != end_cluster:
            cv2.line(im_c, (line[0, 0], line[0, 1]), (line[0,2], line[0,3]), green, 3)
            #cv2.putText(im_c, str(idx), (line[0,0], line[0,1]), font, 1, green, lineType)

#cv2.drawContours(im_c, contours, -1, (0,255,0), 3)

cv2.imwrite("cluster.jpg", im_c)
cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
cv2.imshow('BindingBox',im_c)
cv2.waitKey(0)