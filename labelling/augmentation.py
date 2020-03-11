# Creates a train and test set from directory
# x has shape (-1 x 256 x 256 x 3), range [0, 1], type float32
# y has shape (-1 x 256 x 256), range [0, 1], type float32

import cv2
import os
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# input argument
parser = argparse.ArgumentParser()
parser.add_argument('pathname', type=str, help='Name of labelled data directory')
args = parser.parse_args()

# paths
data_path = os.path.join(args.pathname, 'data')
label_path = os.path.join(args.pathname, 'label')
outfile = 'output.hdf5'

# threshold - above will be white, below will be black
# black(0), grey_8(51), grey_4(153), grey_2(204), grey_1(230), white(255)
threshold = 50

# function to thicken lines
def edit_label(label):
    for x in np.nditer(label, op_flags=['readwrite']):
        if x > threshold:
            x[...] = 255
        else:
            x[...] = 0
    return label


# count number of files
length = 0
for subdir, dirs, files in os.walk(data_path):
    for file in files:
        length += 1

# list of files
data_list = np.zeros((4*length, 256, 256, 3), dtype='float32')
label_list = np.zeros((4*length, 256, 256), dtype='float32')

# augment data and labels in directory
count = 0
for subdir, dirs, files in os.walk(data_path):
    for file in tqdm(files):
        # read data and labels
        # get image from path given
        filename = os.path.join(data_path, file)
        data = cv2.imread(filename)
        # get label from path
        filename = os.path.join(label_path, file)
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label = edit_label(label)

        # augment data
        # flip horizontally
        flipped_data = cv2.flip(data, 1)
        flipped_label = cv2.flip(label, 1)
        # brighten image (no need to augment label)
        bright_data = cv2.convertScaleAbs(data, alpha=1, beta=50)
        # flip + brighten image
        flipped_bright_data = cv2.convertScaleAbs(flipped_data, alpha=1, beta=50)

        data_list[4*count] = data[:, :, ::-1].astype('float32')/255
        data_list[4*count+1] = flipped_data[:, :, ::-1].astype('float32')/255
        data_list[4*count+2] = bright_data[:, :, ::-1].astype('float32')/255
        data_list[4*count+3] = flipped_bright_data[:, :, ::-1].astype('float32')/255
        label_list[4*count] = (label/255)
        label_list[4*count+1] = (flipped_label/255)
        label_list[4*count+2] = (label/255)
        label_list[4*count+3] = (flipped_label/255)

        # # For debugging purposes
        # plt.subplot(2, 2, 1)
        # plt.imshow(data_list[0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(data_list[1])
        # plt.subplot(2, 2, 3)
        # plt.imshow(label_list[2])
        # plt.subplot(2, 2, 4)
        # plt.imshow(label_list[3])
        # plt.show()

        count += 1
print('Data', data_list.dtype, data_list.shape)
print('Label', label_list.dtype, label_list.shape)

# split x and y into train and test sets
print('Performing train-test split...')
SPLIT = 0.7
split_point = int(data_list.shape[0] * SPLIT)
train_x = data_list[0 : split_point]
train_y = label_list[0 : split_point]
test_x = data_list[split_point+1 : data_list.shape[0]-1]
test_y = label_list[split_point+1 : data_list.shape[0]-1]
print(train_x.dtype, train_x.shape)
print(train_y.dtype, train_y.shape)
print(test_x.dtype, test_x.shape)
print(test_y.dtype, test_y.shape)

# mean and standard deviation
mean = np.zeros(3)
for data in tqdm(train_x):
    mean += np.mean(data, axis=(0,1))
mean = mean / train_x.shape[0]
print(mean)
std = np.zeros(3)
for data in tqdm(train_x):
    std += np.power(np.mean(data, axis=(0,1))-mean, 2)
std = np.sqrt(std / train_x.shape[0])
print(std)

# save
print('Saving file...')
# save as hdf5 file
f = h5py.File(outfile, "w")
print('Compressing train images...')
train_images = f.create_dataset("train_images", data = train_x, compression="gzip")
print('Compressing train labels...')
train_labels = f.create_dataset("train_labels", data = train_y, compression="gzip")
print('Compressing test images...')
test_images = f.create_dataset("test_images", data = test_x, compression="gzip")
print('Compressing test labels...')
test_labels = f.create_dataset("test_labels", data = test_y, compression="gzip")
mean = f.create_dataset("mean", data = mean, compression="gzip")
std = f.create_dataset("std", data = std, compression="gzip")
f.close()
# save as npz file
#np.savez(npzfile, train_images=train_x, train_labels=train_y, test_images=test_x, test_labels=test_y)
