# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:30 2019

@author: ASUS
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math as m

### load input image and convert it to grayscale
dir_route = '/Users/Mely/Documents/School/FIMEE/8vo semestre/Computacion Flexible/data/'
img = cv2.imread(dir_route+"9.png")

#def process_inputImage(img):
    
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
###crop edge of each character
_, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
#cv2.imshow('image',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(thresh,'gray')

#### extract all contours
_, contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
###save (x,y,width height) for each contour
bb_list = []
for c in contours:  
    bb = cv2.boundingRect(c)
    # save all boxes except the one that has the exact dimensions of the image (x, y, width, height)
    if (bb[0] == 0 and bb[1] == 0 and bb[2] == img.shape[1] and bb[3] == img.shape[0]):
        continue
    bb_list.append(bb)

   
bb_list.sort(key=lambda x:x[0]) #order bb_list in the order of appearance

img_boxes = img.copy() #a copy of the original image
crop_images=[]
shapes=[]
lenY = []
lenX = []
for bb in bb_list:
   x,y,w,h = bb
   #draw the rectangles of the contours detected by findContours function
   cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
   crop_images.append(thresh[y:y+h, x:x+w])
   shapes.append(crop_images[-1].shape)
   lenY.append(shapes[0])

lenY=[i[0] for i in shapes]
mean_lensY = np.mean(lenY)
std_lensY = np.std(lenY)
lenX=[i[1] for i in shapes if i[0]>mean_lensY]
mean_lensX = np.mean(lenX)
std_lensX = np.std(lenX)

###Filter the images found and risize them

resize_crop_images=[]  
for i in crop_images:
    if i.shape[0]+(std_lensY/2) <mean_lensY: #check if the box has the mean len
        continue
    else:
        if i.shape[1] > std_lensX+ mean_lensX:
            n_parts = m.ceil(i.shape[1]/mean_lensX)
            new_len = int(i.shape[1]/n_parts)
            ##crop
            for j in range(n_parts):
                new_i = i[:,j*new_len:j*new_len+new_len-1]
                resize_crop_images.append(cv2.resize(new_i, dsize=(15, 15), interpolation=cv2.INTER_CUBIC))
        else:
            resize_crop_images.append(cv2.resize(i, dsize=(15, 15), interpolation=cv2.INTER_CUBIC))
    
cv2.imwrite("boxes.jpg", img_boxes)   