# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:04:06 2021

@author: volte
"""


import cv2
import numpy as np
from datetime import datetime
import matplotlib as mat
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import feature


def edgemap(image):
    filt_img = ndimage.gaussian_filter(image, 3)
    edges = feature.canny(filt_img, 2.5)
    #plt.imshow(edges)
    #plt.show()
    return edges

#function to segment only relevant layers
def segmentLayer(image, edges, savelocation):

    cutimg = image
    leny, lenx = image.shape

    layercount = 0
    pixeldist = 0
    guard = False
    start_count_pixel = False
    cut_pixel = np.zeros(leny)

    for col in range(0, leny):
        for row in range(5, lenx):
            if layercount <= 1:
                cutimg[row-5][col] = 0
                cut_pixel[col] = row
                if edges[row][col] == 1 and not guard:
                    layercount += 1
                    start_count_pixel = True
                    guard = True  # skipping next cols (we don't want to look at edges one edge next to the one we found already)
                if pixeldist > 40:
                    guard = False # open guard when we reached 25 pixels below the first edge
            if start_count_pixel:
                pixeldist += 1
        layercount = 0
        pixeldist = 0
        start_count_pixel = False

    for row in range(0, lenx):
        col = lenx - 1
        while edges[col][row] == 0 and col >= 0:
            cutimg[col][row] = 0
            col -= 1


    plt.figure() #save output edges / image
    plt.imshow(cutimg)
    plt.savefig(savelocation)
    #plt.show()


    return cutimg, cut_pixel




##Function to increase brightness and contrast

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf



##function to do blob detection
def blob_detection(image, pars):
    params = pars 
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(image)
   # if not len(keypoints):
    #    return("no blobs detected")
    blank = np.zeros((1, 1)) 
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                            ##formatting the picture 
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    date = datetime. now(). strftime("%Y_%m_%d%I%M%S_%p")
 #   plt.imsave("filename_{}.png".format(date), blobs)
    return(keypoints)
    
##function to calculate distance from layer
def distance_check(photor_img, keypoints):
    ret, thresh = cv2.threshold(photor_img, 127, 255, 0)
    indic=np.argwhere(thresh==255)
    import math
    for i in keypoints:
        try:
            x=int(i.pt[0]-144)
            y=int(i.pt[1]-60)
            radius=int(i.size/2)
        except AttributeError:
            print("no keypoints")
            break
            # print(x, y, radius)
        min_dist=1000
    
        curr_est=[]
        for j in indic:
            x_diff=j[1]-x
            y_diff=j[0]-y
            dist=math.sqrt(x_diff*x_diff + y_diff*y_diff)
            if dist < min_dist:
                min_dist=dist
                curr_est=[j[0], j[1]]
                #print(curr_est, min_dist)
                if min_dist < (radius*1):
                    return(True)

##function combining everything and returning true/false for input images
def test_for_srf(folder_img_name, folder_photor_img_name, params):
    count=0
    list_of_trues_falses=[]
    all_img=os.listdir(folder_img_name)
    all_ref_img=os.listdir(folder_photor_img_name)
    for i in range(0, len(all_img)):
        img=plt.imread("{}/{}".format(folder_img_name, all_img[i]))
        img=cv2.medianBlur(img, 5)
        img=apply_brightness_contrast(img, brightness=0, contrast=60)

        img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = img.astype(np.uint8)
        #plt.imshow(img)

        ref=plt.imread("{}/{}".format(folder_photor_img_name, all_ref_img[i]))
        ref=ref[60:400, 150:500, 0]
        ref = cv2.normalize(ref, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        ref = ref.astype(np.uint8)
        #plt.imshow(ref)
        keys=blob_detection(img, params)
        if distance_check(ref, keys)==True:
            count+=1
            list_of_trues_falses.append(1)
        else:
            list_of_trues_falses.append(0)
    print("percentage SRF:", count/len(all_img)*100)
    return all_img, list_of_trues_falses
    
