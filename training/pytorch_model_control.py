import numpy as np
import h5py
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch import nn


# global variables
filename = 'output.hdf5'    # data array

# Extract data from compressed hdf5
f = h5py.File(filename, "r")
print('Unpacking train labels...')
train_labels = np.array(f.get("train_labels"))
print(train_labels.dtype, train_labels.shape)

# create labels and dummy labels
train_labels = torch.from_numpy(train_labels).view(-1, 256, 256).type(torch.float32)
dummy_label = torch.ones([256, 256], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss()
    
loss_train = 0
for labels in tqdm(train_labels):
    loss = criterion(dummy_label, labels)
    loss_train += loss
loss_train = loss_train/train_labels.shape[0]
print(loss_train)