import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import argparse
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch import nn, optim
from torchsummary import summary

print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'), torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, help='Operation mode (train or test)')
args = parser.parse_args()
if args.mode == 'train':
    print('TRAINING MODEL...')
if args.mode == 'test':
    print('TESTING MODEL...')

# global variables
filename = 'output.hdf5'    # data array
BATCH_SIZE = 1              # batch size
n_epochs = 20               # number of epochs
PATH = 'my_model.pt'        # path of saved model

# model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input of 3 x 256 x 256, output of 64 x 256 x 256
        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )

        # Input of 64 x 256 x 256, output of 128 x 128 x 128
        self.encode2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        # Input of 128 x 128 x 128, output of 256 x 64 x 64
        self.encode3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        # Input of 256 x 64 x 64, output of 512 x 32 x 32
        self.encode4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )

        # Input of 512 x 32 x 32, output of 256 x 64 x 64
        self.decode4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # Input of 512 x 64 x 64, output of 128 x 128 x 128
        self.decode3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        # Input of 256 x 128 x 128, output of 64 x 256 x 256
        self.decode2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        # Input of 128 x 256 x 256, output of 2 x 256 x 256
        self.decode1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )


    # Defining the forward pass
    def forward(self, x):
        x = self.encode1(x)
        x1 = x.clone()
        x = self.encode2(x)
        x2 = x.clone()
        x = self.encode3(x)
        x3 = x.clone()
        x = self.encode4(x)
        x = self.decode4(x)
        x = torch.cat((x3, x), 1)
        x = self.decode3(x)
        x = torch.cat((x2, x), 1)
        x = self.decode2(x)
        x = torch.cat((x1, x), 1)
        x = self.decode1(x)
        return x


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


# creates a mask for display
def create_mask(pred_mask):
    # the label assigned to the pixel is the channel with the highest value
    pred_mask = torch.argmax(pred_mask, dim=0)
    pred_mask = pred_mask.numpy()
    return pred_mask


# show predictions from a dataset
def show_predictions(dataset, num):
    for i, data in enumerate(dataset, 0):
        if i >= num:
            break
        image, mask = data
        with torch.no_grad():
            pred_mask = model(image.cuda())
        image = image.to(torch.device("cpu")).view(-1, 256, 256, 3)
        mask = mask.to(torch.device("cpu"))
        pred_mask = pred_mask.to(torch.device("cpu"))
        print(accuracy(pred_mask[0], mask[0]))
        display([image[0].numpy(), mask[0].numpy(), create_mask(pred_mask[0])])


def accuracy(pred, true):
    sum = 0
    pred = create_mask(pred)
    for x in range(256):
        for y in range(256):
            if pred[x][y] == true[x][y]:
                sum += 1
    return sum/(256*256)


# Train mode
if args.mode == 'train':
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
    train_y = train_y.view(-1, 256, 256).type(torch.LongTensor)
    val_x = torch.from_numpy(test_images)
    val_x = val_x.view(-1, 3, 256, 256)
    val_y = torch.from_numpy(test_labels)
    val_y = val_y.view(-1, 256, 256).type(torch.LongTensor)
    print(train_x.dtype, train_x.shape)
    print(train_y.dtype, train_y.shape)

    # create dataset
    trainset = torch.utils.data.TensorDataset(train_x, train_y)
    testset = torch.utils.data.TensorDataset(val_x, val_y)

    # defining the model
    model = Net()
    model = model.cuda()
    # defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    summary(model, (3, 256, 256))

    # empty list to store training and validation losses
    train_losses = []
    val_losses = []
    train_size = train_images.shape[0]
    val_size = test_images.shape[0]

    # training the model
    # training the model
    for epoch in range(n_epochs):
        print('Epoch ',epoch+1, ' of ', n_epochs, ': ')
        trainloader=torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader=torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        loss_train = 0
        loss_val = 0
        # computing the training loss batch-wise
        for inputs, labels in tqdm(trainloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda().type(torch.cuda.LongTensor))
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            # prediction for training and validation set
            output_train = model(inputs)
            # computing the training and validation loss
            loss = criterion(output_train, labels) # compare the likelihood of a pixel being 1
            # computing the updated weights of all the model parameters
            loss.backward()
            optimizer.step()
            loss_train += loss
        loss_train = loss_train / (train_size/BATCH_SIZE)

        # computing the validation loss batch-wise
        for inputs, labels in tqdm(testloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda().type(torch.cuda.LongTensor))
            with torch.no_grad():
                output_val = model(inputs)
            loss_val += criterion(output_val, labels)
        loss_val = loss_val / (val_size/BATCH_SIZE)

        # Update values and print to screen
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print('train_loss :', loss_train, '\t', 'val_loss :', loss_val)

    # save model
    torch.save(model.state_dict(), PATH)

    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()


# Test mode
if args.mode == 'test':
    # Unpack test sample
    f = h5py.File(filename, "r")
    print('Unpackaging test images...')
    test_images = np.array(f.get("test_images"))
    print('Unpackaging test labels...')
    test_labels = np.array(f.get("test_labels"))

    # generate dataset
    val_x = torch.from_numpy(test_images)
    val_x = val_x.view(-1, 3, 256, 256)
    val_y = torch.from_numpy(test_labels)
    val_y = val_y.view(-1, 256, 256)
    print(val_x.shape, val_x.type)
    testset = torch.utils.data.TensorDataset(val_x, val_y)
    testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # Load model
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model = model.cuda()
    summary(model, (3, 256, 256))

    # visualise sample test data
    show_predictions(testloader, 5)
