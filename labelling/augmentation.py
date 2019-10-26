import cv2
import os
import numpy as np

# paths
data_path = 'labelled_data/data'
label_path = 'labelled_data/label'

# global variables
count = len(os.listdir(data_path))
data_list = []
label_list = []
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
        count += 1
        # write brightened images
        out_filename = str(count) + '.jpg'
        out_data_path = os.path.join(data_path, out_filename)
        out_label_path = os.path.join(label_path, out_filename)
        cv2.imwrite(out_data_path, bright_data)
        cv2.imwrite(out_label_path, label)
        count += 1
        # write flipped + brightened images
        out_filename = str(count) + '.jpg'
        out_data_path = os.path.join(data_path, out_filename)
        out_label_path = os.path.join(label_path, out_filename)
        cv2.imwrite(out_data_path, flipped_bright_data)
        cv2.imwrite(out_label_path, flipped_label)
        count += 1

        # Add to list
        data_list.append(data)
        data_list.append(flipped_data)
        data_list.append(bright_data)
        data_list.append(flipped_bright_data)
        label_list.append(label)
        label_list.append(flipped_label)
        label_list.append(label)
        label_list.append(flipped_label)

        if count % 1000 == 0:
            print('Finished', count, 'images')

SPLIT = 0.7
# split x and y into train and test sets
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
np.savez('labelled_data/outfile', train_images=train_x, train_labels=train_y, test_images=test_x, test_labels=test_y)
