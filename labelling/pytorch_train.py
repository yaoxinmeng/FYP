import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import Adam, SGD
from torchsummary import summary


print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'), torch.cuda.get_device_name(0))

# global variables
filename = 'output_alt.hdf5'    # data array
BATCH_SIZE = 2              # batch size
n_epochs = 20               # number of epochs

# model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # takes in a RGB input of 256x256, output filters = 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(128),

            # second convolution layer, output filters = 64
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(64),

            # third convolution layer, output filters = 128
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(32),

            # Upsampling using ConvTranspose2d, output filters = 64
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Upsampling using ConvTranspose2d, output filters = 32
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Upsampling using ConvTranspose2d, output filters = 1
            nn.ConvTranspose2d(32, 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(2),
            # nn.Softmax2d(),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(-1, 256, 256)
        return x


# Extract data from compressed hdf5
f = h5py.File(filename, "r")
print('Unpacking train images...')
train_images = np.array(f.get("train_images"))
print('Unpacking train labels...')
train_labels = np.array(f.get("train_labels"))
print('Unpacking test images...')
test_images = np.array(f.get("test_images"))
print('Unpacking test labels...')
test_labels = np.array(f.get("test_labels"))
print(train_images.dtype, train_images.shape)
print(train_labels.dtype, train_labels.shape)

# Convert to torch format
train_x = torch.from_numpy(train_images)
train_x = train_x.view(-1, 3, 256, 256)
train_y = torch.from_numpy(train_labels)
train_y = train_y.view(-1, 256, 256)
val_x = torch.from_numpy(test_images)
val_x = val_x.view(-1, 3, 256, 256)
val_y = torch.from_numpy(test_labels)
val_y = val_y.view(-1, 256, 256)
print(train_x.dtype, train_x.shape)
print(train_y.dtype, train_y.shape)

# create dataset
trainset = torch.utils.data.TensorDataset(train_x, train_y)
testset = torch.utils.data.TensorDataset(val_x, val_y)

# defining the model
model = Net()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
summary(model, (3, 256, 256))

# empty list to store training and validation losses
train_losses = []
val_losses = []
train_size = train_images.shape[0]
val_size = test_images.shape[0]

# training the model
for epoch in range(n_epochs):
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader=torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    loss_train = 0
    loss_val = 0
    # computing the training loss batch-wise
    for inputs, labels in tqdm(trainloader):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        # prediction for training and validation set
        output_train = model(inputs)
        # computing the training and validation loss
        loss = criterion(output_train, labels.type(torch.cuda.LongTensor)) # compare the likelihood of a pixel being 1
        # computing the updated weights of all the model parameters
        loss.backward()
        optimizer.step()
        loss_train += loss
    loss_train = loss_train / train_size

    # computing the validation loss batch-wise
    for inputs, labels in tqdm(testloader):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        with torch.no_grad():
            output_val = model(inputs)
        loss_val += criterion(output_val, labels.type(torch.cuda.LongTensor))
    loss_val = loss_val / val_size

    # Update values and print to screen
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    print('Epoch : ',epoch+1, '\t', 'train_loss :', loss_train, '\t', 'val_loss :', loss_val)


# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'my_model.pt')
