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
BATCH_SIZE = 2              # batch size
n_epochs = 50               # number of epochs
PATH = 'my_model_u.pt'      # path of saved model
FIG_NAME = 'u_loss.png'     # name of figure plot


# encoder submodel definition
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(C_out, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(C_out, track_running_stats=False),
            nn.LeakyReLU(),
        )
    
    # Defining the forward pass
    def forward(self, x):
        x = self.conv(x)
        return x
        
        
# model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input of 3 x 256 x 256, output of 64 x 256 x 256
        self.encode1 = Conv(3, 64)
        
        # Input of 64 x 256 x 256, output of 128 x 128 x 128
        self.encode2 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(64, 128)
        )

        # Input of 128 x 128 x 128, output of 256 x 64 x 64
        self.encode3 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(128, 256)
        )
        
        # Input of 256 x 64 x 64, output of 512 x 32 x 32
        self.encode4 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(256, 512)
        )
        
        # Input of 512 x 32 x 32, output of 256 x 64 x 64
        self.decode4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        # Input of 512 x 64 x 64, output of 128 x 128 x 128
        self.decode3 = nn.Sequential(
            Conv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        
        # Input of 256 x 128 x 128, output of 64 x 256 x 256
        self.decode2 = nn.Sequential(
            Conv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        # Input of 128 x 256 x 256, output of 2 x 256 x 256
        self.decode1 = nn.Sequential(
            Conv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
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
        x = x.view(-1, 256, 256)
        return x
    
    # Show the filters from each layer
    def visualize_features(self, x):
        plt.figure(1)
        plt.imshow(x[0].cpu().permute(1, 2, 0))
        x = self.encode1(x)
        for i in range(64):
            plt.figure(2)
            plt.subplot(8, 8, i+1)
            plt.imshow(x[0,i,:,:].cpu())
            plt.axis('off')
        x = self.encode2(x)
        for i in range(128):
            plt.figure(3)
            plt.subplot(8, 16, i+1)
            plt.imshow(x[0,i,:,:].cpu())
            plt.axis('off')
        x = self.encode3(x)
        for i in range(256):
            plt.figure(4)
            plt.subplot(16, 16, i+1)
            plt.imshow(x[0,i,:,:].cpu())
            plt.axis('off')
        x = self.encode4(x)
        for i in range(512):
            plt.figure(5)
            plt.subplot(16, 32, i+1)
            plt.imshow(x[0,i,:,:].cpu())
            plt.axis('off')
        plt.show()

    
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
    testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    # Load model
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model = model.cuda()
    model.eval()
    summary(model, (3, 256, 256))
    
    # visualise features
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        with torch.no_grad():
            model.visualize_features(inputs.cuda())
        break
    
    # visualise sample test data
    show_predictions(testloader, 5)
    