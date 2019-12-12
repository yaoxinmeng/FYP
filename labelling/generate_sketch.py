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

# extract file as array and manipulate array data
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


# apply Hough Transform on processed label
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


# get equations of each line in image and attempt to join them
# polar equation of line is rho = d/(cos(theta-beta)), where
# a point on the line is (rho, theta)
def getLines(label):
    d_range = int(sqrt(2)*256)
    beta_range = 360
    param = np.zeros((d_range, beta_range))
    with np.nditer(param, op_flags=['readwrite']) as temp:
        for x in range(256):
            for y in range(256):
                rho = sqrt(x*x + y*y)
                theta = arctan(y/x)
                for d in range(d_range):
                    for beta in range(beta_range):
                        beta_rad = beta/180 * np.pi
                        match = d / (rho * cos(theta - beta_rad))
                        if match >= 0.99:
                            temp[d, beta] = temp[d, beta] + 1



parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Name of input label file')
args = parser.parse_args()

label = read_file(args.filename)
houghtransform(label)
