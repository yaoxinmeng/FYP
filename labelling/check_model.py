import tensorflow as tf
import numpy as np
import cv2
import h5py
from tensorflow.keras import optimizers


# Custom loss function
def mse_loss(y_pred, y_true):
  error = tf.math.squared_difference(y_pred, y_true)
  sum = tf.reduce_sum(error)
  return sum/65536


# Load model
new_model = tf.keras.models.load_model('my_model.hdf5', compile=False)
new_model.summary()

# Unpack test sample
f = h5py.File('output.hdf5', "r")
print('Unpackaging test images...')
test_images = np.array(f.get("test_images"))
print('Unpackaging test labels...')
test_labels = np.array(f.get("test_labels"))
test_labels = np.float32(test_labels/255.0)

# Evaluate new model
adam = optimizers.Adam(learning_rate=0.0001, epsilon=0.1)
new_model.compile(optimizer=adam, loss=mse_loss)
acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
