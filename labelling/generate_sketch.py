import cv2
import os
import numpy as np
import argparse

# threshold value for pixel to qualify as a label
threshold = 0.1
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
    cv2.imshow('label', label)
    label = np.array(label).astype('float32')
    label =  label/ 255.0
    ################################################

    # categorise labels into either black (0) or white (1), and
    # inverse white and black for Hough Transform later
    with np.nditer(label, op_flags=['readwrite']) as temp:
        for point in temp:
            if point <= threshold:
                point[...] = 1
            else:
                point[...] = 0
    return np.uint8(label * 255.0)


def houghtransform(label):
    img = np.copy(label)*0 + 255
    lines = cv2.HoughLinesP(label, rho, theta, houghthreshold, minLineLength, maxLineGap)
    if lines.any():
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('edges', label)
    cv2.imshow('hough', img)
    cv2.waitKey(0)
    cv2.imwrite('houghlines5.jpg',img)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Name of input label file')
args = parser.parse_args()

label = read_file(args.filename)
houghtransform(label)
