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
        label = cv2.imread(filename)

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

data_list = np.array(data_list)
label_list = np.array(label_list)
np.savez('outfile', data=data_list, label=label_list)
