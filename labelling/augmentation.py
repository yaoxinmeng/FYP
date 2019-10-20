import cv2
import os

# paths
data_path = 'labelled_data/data'
label_path = 'labelled_data/label'

# global variables
count = len(os.listdir(data_path))

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

        if count % 1000 == 0:
            print('Finished', count, 'images')
