import h5py
import numpy as np

f = h5py.File("output.hdf5", "r")
train_images = f.get("train_images")
n1 = np.array(train_images)
