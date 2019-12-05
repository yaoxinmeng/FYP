import cv2
import os
import numpy as np
import argparse

# threshold value for pixel to qualify as a label
threshold = 0.8
# Hough Transform parameters
minLineLength = 20
maxLineGap = 5
rho = 1
theta = np.pi/180
houghthreshold = 15

# extract file as array
def read_file(file):
    # Comment out this section as necessary
    ################################################
    label = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    label = np.array(label)
    #label.reshape(256, 256, 1).astype('float32')
    label =  label/ 255.0
    ################################################
    for points in label:
        for point in points:
            if point <= threshold:
                point = 0
            else:
                point = 1
    cv2.imshow('label', label)
    cv2.waitKey(0)
    return label * 255.0


def houghtransform(label):
    img = np.copy(label)*0 + 255
    edges = cv2.Canny(label, 50, 150, apertureSize = 3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, rho, theta, houghthreshold, minLineLength, maxLineGap)
    if lines.any():
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('canny', edges)
    cv2.imshow('hough', img)
    cv2.waitKey(0)
    cv2.imwrite('houghlines5.jpg',img)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Name of input label file')
args = parser.parse_args()

label = read_file(args.filename)
houghtransform(label)
