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
import torchvision.transforms as T

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
BATCH_SIZE = 3              # batch size
n_epochs = 50               # number of epochs
PATH = 'my_model_dense.pt'  # path of saved model
FIG_NAME = 'dense_loss.png' # name of figure plot


# Dense block submodel definition (output is 4*k)
class DenseBlock(nn.Module):
    def __init__(self, C_in, k):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(C_in, k, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(k, track_running_stats=False),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(C_in+k, k, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(k, track_running_stats=False),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(C_in+2*k, k, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(k, track_running_stats=False),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(C_in+3*k, k, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(k, track_running_stats=False),
            nn.ReLU(),
        )

    # Defining the forward pass
    def forward(self, x):
        x1 = self.layer1(x)
        x = torch.cat((x, x1), 1)
        x2 = self.layer2(x)
        x = torch.cat((x, x2), 1)
        x3 = self.layer3(x)
        x = torch.cat((x, x3), 1)
        x4 = self.layer4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        return x


# Transition down module (output is C_in)
class TD(nn.Module):
    def __init__(self, C_in):
        super(TD, self).__init__()
        self.conv = nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)

    # Defining the forward pass
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


# Transition up module (output is C_in/2)
class TU(nn.Module):
    def __init__(self, C_in):
        super(TU, self).__init__()
        self.conv_t = nn.ConvTranspose2d(C_in, int(C_in/2), kernel_size=2, stride=2)

    # Defining the forward pass
    def forward(self, x):
        x = self.conv_t(x)
        return x


# DenseNet model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Output of 64 channels
        self.firstconv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Output of 64 channels
        self.block1 = DenseBlock(64, 16)

        # Output of 128 channels
        self.td = TD(128)
        self.block2 = DenseBlock(128, 32)

        # Output of 256 channels
        self.bottleneck = nn.Sequential(
            TD(256),
            DenseBlock(256, 128),
            TU(512)
        )

        # Output of 128 channels
        self.block3 = DenseBlock(512, 64)
        self.tu = TU(256)

        # Output of 64 channels
        self.block4 = DenseBlock(256, 16)

        # Output of 2 channels
        self.finalconv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    # Defining the forward pass
    def forward(self, x):
        x = self.firstconv(x)
        x1 = self.block1(x)
        x1 = torch.cat((x, x1), 1)
        x = self.td(x1)
        x2 = self.block2(x)
        x2 = torch.cat((x, x2), 1)
        x = self.bottleneck(x2)
        x = torch.cat((x, x2), 1)
        x = self.block3(x)
        x = self.tu(x)
        x = torch.cat((x, x1), 1)
        x = self.block4(x)
        x = self.finalconv(x)
        x = x.view(-1, 256, 256)
        return x


# Custom tensor dataset with support of transforms
class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays
        self.image_trf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        inplace = True)
            ])

    def __getitem__(self, index):
        x = self.arrays[0][index]
        x = self.image_trf(x)
        y = self.arrays[1][index]
        y = torch.from_numpy(y).view(256, 256).type(torch.float32)
        return x, y

    def __len__(self):
        return self.arrays[0].shape[0]


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


# # creates a mask for display
# def create_mask(pred_mask):
    # # the label assigned to the pixel is the channel with the highest value
    # pred_mask = torch.argmax(pred_mask, dim=0)
    # return pred_mask


# show predictions from a dataset
def show_predictions(dataset, num):
    activation = nn.Sigmoid()
    for i, data in enumerate(dataset, 0):
        if i >= num:
            break
        image, mask = data
        with torch.no_grad():
            pred_mask = model(image.cuda())
            pred_mask = activation(pred_mask)
        pred_mask = pred_mask[0].cpu()
        mask = mask[0]
        image = image[0]
        print(accuracy(pred_mask, mask))
        display([image.permute(1, 2, 0), mask, pred_mask])


def accuracy(pred, true):
    sum = 0
    for x in range(256):
        for y in range(256):
            sum += abs(pred[x][y] - true[x][y])
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

    # create dataset
    trainset = CustomTensorDataset((train_images, train_labels))
    testset = CustomTensorDataset((test_images, test_labels))

    # # Visualize sample from dataset
    # x, y = trainset[0]
    # print(x.dtype, x.shape)
    # print(y.dtype, y.shape)
    # plt.subplot(1, 3, 1)
    # plt.imshow(train_images[0])
    # plt.subplot(1, 3, 2)
    # plt.imshow(x.permute(1, 2, 0))
    # plt.subplot(1, 3, 3)
    # plt.imshow(y)
    # plt.show()
    # del x, y

    # defining the model
    model = Net()
    model = model.cuda()
    # defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = nn.BCEWithLogitsLoss()
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
        model.train()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
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
        model.eval()
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                output_val = model(inputs)
            loss_val += criterion(output_val, labels)
        loss_val = loss_val / (val_size/BATCH_SIZE)

        # Update values and print to screen
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print('train_loss :', loss_train, '\t', 'val_loss :', loss_val)

    # save model
    print('Saving...')
    torch.save(model.state_dict(), PATH)

    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig(FIG_NAME)


# Test mode
if args.mode == 'test':
    # Unpack test sample
    f = h5py.File(filename, "r")
    print('Unpackaging test images...')
    test_images = np.array(f.get("test_images"))
    print('Unpackaging test labels...')
    test_labels = np.array(f.get("test_labels"))

    # generate dataset
    testset = CustomTensorDataset((test_images, test_labels))
    testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # Load model
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model = model.cuda()
    model.eval()
    summary(model, (3, 256, 256))

    # evaluate loss adn accuracy
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    activation = nn.Sigmoid()
    activation = activation.cuda()
    loss = 0
    acc = 0
    for inputs, labels in tqdm(testloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            output_val = model(inputs)
        loss += criterion(output_val, labels)

        output_val = activation(output_val)
        acc += accuracy(output_val[0], labels[0])
    loss = loss / test_images.shape[0]
    acc = acc / test_images.shape[0]
    print('loss :', loss, '\t', 'acc :', acc)

    # visualise sample test data
    show_predictions(testloader, 5)
