import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch import nn, optim
from torchsummary import summary
from PIL import Image
import torchvision.transforms as T
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'), torch.cuda.get_device_name(0))

PATH = 'my_model_u.pt'  # path of saved model
PATH_1 = 'my_model_u_1.pt'  # path of saved model
PATH_ORI = 'my_model_u_ori.pt'  # path of saved model
threshold = 0.5             # threshold for a pixel to be considered an edge
                            # black is <= threshold

# convert image to input for model
img_transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                inplace = True)
    ])

# function to thicken lines
def edit_label(label):
    for x in np.nditer(label, op_flags=['readwrite']):
        if x > threshold:
            x[...] = 1
        else:
            x[...] = 0
    return label

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


# Load model
model = Net()
model.load_state_dict(torch.load(PATH))
model = model.cuda()
model.eval()
summary(model, (3, 256, 256))

model_1 = Net()
model_1.load_state_dict(torch.load(PATH_1))
model_1 = model_1.cuda()
model_1.eval()
summary(model_1, (3, 256, 256))

model_ori = Net()
model_ori.load_state_dict(torch.load(PATH_ORI))
model_ori = model_ori.cuda()
model_ori.eval()
summary(model_ori, (3, 256, 256))

# read image from PATH
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path of image file')
args = parser.parse_args()
im = Image.open(args.image_path)

# run model
input = img_transform(im).view(-1, 3, 256, 256)
activation = nn.Sigmoid()
with torch.no_grad():
    mask = model(input.cuda())
    mask = activation(mask)
    mask1 = model_1(input.cuda())
    mask1 = activation(mask1)
    mask_ori = model_ori(input.cuda())
    mask_ori = activation(mask_ori)

# Prepare data to be displayed
input = input[0].permute(1, 2, 0).cpu()
mask = np.array(mask[0].cpu())
mask = edit_label(mask)
mask1 = np.array(mask1[0].cpu())
mask1 = edit_label(mask1)
mask_ori = np.array(mask_ori[0].cpu())
mask_ori = edit_label(mask_ori)

# show results on screen
plt.figure(figsize=(15, 15))
title = ['Input Image', 'Normalised Image', 'Predicted Mask (1-pixel)', 'Predicted Mask (3-pixel)', 'Predicted Mask (heatmap)']
display_list = [im, input, mask1, mask, mask_ori]
for i in range(len(display_list)):
    plt.subplot(2, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
plt.show()
