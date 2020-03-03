import numpy as np
import h5py
import matplotlib.pyplot as plt

# paths
f = h5py.File('output1.hdf5', "r")
test_labels1 = np.array(f.get("test_labels"))
f = h5py.File('output.hdf5', "r")
test_labels3 = np.array(f.get("test_labels"))
f = h5py.File('output_ori.hdf5', "r")
test_labels = np.array(f.get("test_labels"))

# For debugging purposes
plt.subplot(1, 3, 1)
plt.title('1 pixel')
plt.imshow(test_labels1[0])
plt.subplot(1, 3, 2)
plt.title('3 pixels')
plt.imshow(test_labels3[0])
plt.subplot(1, 3, 3)
plt.title('Heatmap')
plt.imshow(test_labels[0])
plt.show()
