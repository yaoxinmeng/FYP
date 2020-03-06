import numpy as np
import h5py
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

# global variables
filename = 'output.hdf5'    # data array

# visual display of results
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def predict(images, labels, num):
    for i, image in enumerate(images, 0):
        if i >= num:
            break
        edge = cv2.Canny(image, 100, 200)
        display([image, labels[i], edge])


# Extract data from compressed hdf5
f = h5py.File(filename, "r")
print('Unpackaging test images...')
test_images = np.array(f.get("test_images"))
print('Unpackaging test labels...')
test_labels = np.array(f.get("test_labels"))

predict(test_images, test_labels, 2)
