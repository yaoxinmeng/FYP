import cv2
import os
import numpy as np
import argparse

# threshold value for pixel to qualify as a label
threshold = 0.8
parallel = 0.95

# extract file as array
def read_file(file):
    # Comment out this section as necessary
    ################################################
    label = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    label = np.array(label)
    label.reshape(train_size, 256, 256, 1).astype('float32')
    label =  label/ 255.0
    ################################################
    for points in label:
        if points <= threshold:
            points = 0
        else:
            points = 1
    return label


# determine if two vectors are roughly parallel
def isparallel(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    if dot > parallel || dot < -parallel:
        return 1
    else:
        return 0


# count the number of tagged pixels around a given pixel
def neighbours(array, x, y):
    count = 0
    for u in range(x-1, x+2):
        for v in range(y-1, y+2):
            if array[u][v] == 1 && u!= x && v != y:
                count = count + 1
    return count


def keypoints(array):
    points = []
    line_array = np.zeros(256, 256)
    for x in range(0, 255):
        for y in range(0, 255):
            if array[x][y] == 1 && line[x][y] != 1:
                # add as endpoint to points list
                points = points.append(tuple((x,y)))
                # mark out all other points that form the line in line_array
                line_array = trace (array, x, y)
    return points


def trace(array, x, y):
    if neighbours(array, x, y) != 1:


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Name of input label file')
args = parser.parse_args()

read_file(args.filename)
