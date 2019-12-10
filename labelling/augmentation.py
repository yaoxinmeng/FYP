import cv2
import os
import numpy as np
import argparse

# input argument
parser = argparse.ArgumentParser()
parser.add_argument('pathname', type=str, help='Name of labelled data directory')
args = parser.parse_args()

# paths
data_path = os.path.join(args.pathname, 'data')
label_path = os.path.join(args.pathname, 'label')
npzfile = os.path.join(args.pathname, 'outfile')

# get biggest filename in data_path
count = 0
for subdir, dirs, files in os.walk(data_path):
    for file in files:
        temp = file.split('.')
        temp = int(temp[0])
        if temp > count:
            count = temp
count = count + 1


# list of files
data_list = []
label_list = []

# augment data and labels in directory
for subdir, dirs, files in os.walk(data_path):
    for file in files:
        # read data and labels
        # get image from path given
        filename = os.path.join(data_path, file)
        data = cv2.imread(filename)
        # get label from path
        filename = os.path.join(label_path, file)
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # augment data
        # flip horizontally
        flipped_data = cv2.flip(data, 1)
        flipped_label = cv2.flip(label, 1)
        # brighten image (no need to augment label)
        bright_data = cv2.convertScaleAbs(data, alpha=1, beta=50)
        # flip + brighten image
        flipped_bright_data = cv2.convertScaleAbs(flipped_data, alpha=1, beta=50)

        # write to output path
        # write flipped images
        out_filename = str(count) + '.jpg'
        out_data_path = os.path.join(data_path, out_filename)
        out_label_path = os.path.join(label_path, out_filename)
        cv2.imwrite(out_data_path, flipped_data)
        cv2.imwrite(out_label_path, flipped_label)

        # write brightened images
        out_filename = str(count) + '.jpg'
        out_data_path = os.path.join(data_path, out_filename)
        out_label_path = os.path.join(label_path, out_filename)
        cv2.imwrite(out_data_path, bright_data)
        cv2.imwrite(out_label_path, label)

        # write flipped + brightened images
        out_filename = str(count) + '.jpg'
        out_data_path = os.path.join(data_path, out_filename)
        out_label_path = os.path.join(label_path, out_filename)
        cv2.imwrite(out_data_path, flipped_bright_data)
        cv2.imwrite(out_label_path, flipped_label)

        # Add to list
        data_list.append(data)
        data_list.append(flipped_data)
        data_list.append(bright_data)
        data_list.append(flipped_bright_data)
        label_list.append(label)
        label_list.append(flipped_label)
        label_list.append(label)
        label_list.append(flipped_label)

        count += 1
        if count % 1000 == 0:
            print('Finished', count, 'images')

# split x and y into train and test sets
SPLIT = 0.7
split_point = int(len(data_list) * SPLIT)
train_x = data_list[0 : split_point]
train_y = label_list[0 : split_point]
test_x = data_list[split_point+1 : len(data_list)]
test_y = label_list[split_point+1 : len(label_list)]

# convert into array and reshape
train_size = len(train_x)
test_size = len(test_x)
train_x = np.array(train_x)
train_x.reshape(train_size, 256, 256, 3).astype('float32')
train_y = np.array(train_y)
train_y.reshape(train_size, 256, 256, 1).astype('float32')
test_x = np.array(test_x)
test_x.reshape(test_size, 256, 256, 3).astype('float32')
test_y = np.array(test_y)
test_y.reshape(test_size, 256, 256, 1).astype('float32')

# save as npz file
np.savez(npzfile, train_images=train_x, train_labels=train_y, test_images=test_x, test_labels=test_y)
