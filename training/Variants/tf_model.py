import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import cv2
import h5py
from tensorflow.keras import layers, optimizers
from tqdm import tqdm
from matplotlib import pyplot as plt

# global configs, check for GPU and allow memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# global variables
filename = 'output.hdf5'    # data array
BATCH_SIZE = 5              # batch size
SHUFFLE_BUFFER_SIZE = 1000  # should be size of dataset for perfect shuffle


# model generator
def generate_model():
  model = tf.keras.models.Sequential()

  # takes in a RGB input of 256x256, output filters = 64
  model.add(layers.Conv2D(64, (5,5), strides=(1,1), padding='same', input_shape=(256,256,3)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  #model.add(layers.Dropout(0.3))   # randomly remove some neurons
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  assert model.output_shape == (None, 128, 128, 64)   # check output shape

  # second convolution layer, output filters = 128
  model.add(layers.Conv2D(128, (5,5), strides=(1,1), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  #model.add(layers.Dropout(0.3))   # randomly remove some neurons
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  assert model.output_shape == (None, 64, 64, 128)   # check output shape

  # third convolution layer, output filters = 128
  model.add(layers.Conv2D(128, (5,5), strides=(1,1), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  #model.add(layers.Dropout(0.3))   # randomly remove some neurons
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  assert model.output_shape == (None, 32, 32, 128)   # check output shape
    
  # Upsampling using Conv2DTranspose
  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 64, 64, 128)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  #model.add(layers.Dropout(0.3))

  # Upsampling using Conv2DTranspose
  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 64)   # check output shape
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  #model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 256, 256, 1)    # check output shape
  model.add(layers.Activation('softmax'))

  return model


# Custom loss function
def mse_loss(y_pred, y_true):
  error = tf.math.squared_difference(y_pred, y_true)
  sum = tf.reduce_sum(error)
  return sum/65536


# Extract data from compressed hdf5
f = h5py.File(filename, "r")
print('Unpacking train images...')
train_images_temp = np.array(f.get("train_images"))
print('Unpacking train labels...')
train_labels = np.array(f.get("train_labels"))
print('Unpacking test images...')
test_images_temp = np.array(f.get("test_images"))
print('Unpacking test labels...')
test_labels = np.array(f.get("test_labels"))

# Normalize data (train_images and test_images must be done element-wise due to size of array)
print('Normalizing labels...')
train_labels, test_labels = np.float32(train_labels)/255.0, np.float32(test_labels)/255.0
print('Normalizing train images...')
# with np.nditer(train_images, op_flags=['readwrite']) as temp:
   # for point in tqdm(temp):
        # point[...] = np.float32(point)/255.0
BATCH = 1000
length = train_images_temp.shape[0]
total = int(length/BATCH)
train_images = np.zeros(train_images_temp.shape)
for i in tqdm(range(total)):
    if (i+1)*BATCH > length:
        temp = train_images_temp[i*BATCH:length]
        train_images[i*BATCH:length] = np.float32(temp)
    else:
        temp = train_images_temp[i*BATCH:(i+1)*BATCH]
        train_images[i*BATCH:(i+1)*BATCH] = np.float32(temp)
print('Normalizing test images...')
# with np.nditer(test_images, op_flags=['readwrite']) as temp:
   # for point in tqdm(temp):
        # point[...] = np.float32(point)/255.0
length = test_images_temp.shape[0]
total = int(length/BATCH)
test_images = np.zeros(test_images_temp.shape)
for i in tqdm(range(total)):
    if (i+1)*BATCH > length:
        temp = test_images_temp[i*BATCH:length]
        test_images[i*BATCH:length] = np.float32(temp)
    else:
        temp = test_images_temp[i*BATCH:(i+1)*BATCH]
        test_images[i*BATCH:(i+1)*BATCH] = np.float32(temp)      

# Generate dataset
# print('Verify dataset shape:')
# images, labels = next(datagen.flow(train_images, train_labels, BATCH_SIZE))
# print(images.dtype, images.shape)
# print(labels.dtype, labels.shape)
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_dataset = tf.data.Dataset.from_generator(
    # datagen.flow, args=[train_images, train_labels, BATCH_SIZE],
    # output_types=(tf.float32, tf.float32),
    # output_shapes=([None,256,256,3], [None,256,256])
# )
# test_dataset = tf.data.Dataset.from_generator(
    # datagen.flow, args=[test_images, test_labels, BATCH_SIZE],
    # output_types=(tf.float32, tf.float32),
    # output_shapes=([None,256,256,3], [None,256,256])
# )

# # Batch the data
# ds_counter = tf.data.Dataset.from_generator(gen_train, output_types=tf.float32, output_shapes = (), )
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Create neural network and train it
print('Generating model...')
model = generate_model()
model.summary()
adam = optimizers.Adam(learning_rate=0.0001, epsilon=0.1)   # tweak values to optimize training
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)  # validate using test data
model.save('my_model.hdf5')

# Get stats from history object
history_dict = history.history
print(history_dict.keys())    # get keys for history object
acc = history.history['mean_squared_error']            # Train data accuracy
val_acc = history.history['val_mean_squared_error']    # Test data accuracy
epochs = range(len(acc))

# Plot accuracy vs epochs
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, color='blue', label='Train')
plt.plot(epochs, val_acc, color='red', label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
