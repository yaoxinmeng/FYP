from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import cv2
from tensorflow.keras import layers
from IPython.display import clear_output
from matplotlib import pyplot as plt

# paths
data_path = 'labelled_data/data'
label_path = 'labelled_data/label'
SPLIT = 0.7


# FUNCTIONS
def pair_data_label(data_path, label_path):
    x = []
    y = []
    # read data and labels
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            # get image from path given
            filename = os.path.join(data_path, file)
            data = cv2.imread(filename) / 255.0
            # get label with identical name from path
            filename = os.path.join(label_path, file)
            label = cv2.imread(filename) / 255.0

            # add arrays to list
            x.append(data)
            y.append(label)

    # split x and y into train and test sets
    split_point = int(len(x) * SPLIT)
    train_x = x[0 : split_point]
    train_y = y[0 : split_point]
    test_x = x[split_point+1 : len(x)]
    test_y = y[split_point+1 : len(y)]
    return (train_x, train_y), (test_x, test_y)


def generate_model():
  model = tf.keras.models.Sequential()

  # takes in a RGB input of 256x256, output filters = 64
  model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[256,256,3]))
  assert model.output_shape == (None, 128, 128, 64)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))   # randomly remove some neurons
  #model.add(layers.MaxPool2D(pool_size=(2,2)))

  # second convolution layer, output filters = 128
  model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
  assert model.output_shape == (None, 64, 64, 128)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))   # randomly remove some neurons
  #model.add(layers.MaxPool2D(pool_size=(2,2)))

  #model.add(layers.Flatten())

  # Upsampling using Conv2DTranspose
  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 64, 64, 128)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  # Upsampling using Conv2DTranspose
  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 64)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='softmax'))
  assert model.output_shape == (None, 256, 256, 1)    # check output shape

  return model


# MAIN FUNCTION
def main():
    # Get data
    (train_images, train_labels), (test_images, test_labels) = pair_data_label(data_path, label_path)
    print(train_images[0].shape)
    # Create neural network and train it
    model = generate_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, validation_data=(test_images,test_labels), epochs=20)  # validate using test data
    # Get stats from history object
    history_dict = history.history
    print(history_dict.keys())    # get keys for history object
    acc = history.history['accuracy']            # Train data accuracy
    val_acc = history.history['val_accuracy']    # Test data accuracy
    epochs = range(len(acc))
    # Plot accuracy vs epochs
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, color='blue', label='Train')
    plt.plot(epochs, val_acc, color='red', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


if __name__ == '__main__' :
    main()
