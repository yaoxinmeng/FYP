import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
# PyTorch libraries and modules
import torch
from torch import nn, optim
from torchsummary import summary
from PIL import Image
import torchvision.transforms as T
import cv2
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'), torch.cuda.get_device_name(0))

PATH = 'my_model_u.pt'  # path of saved model
threshold = 0.5             # threshold for a pixel to be considered an edge
                            # black is <= threshold

# convert image to input for model
img_transform = T.ToTensor()

# function to thicken lines
def edit_label(label):
    for x in np.nditer(label, op_flags=['readwrite']):
        if x > threshold:
            x[...] = 0
        else:
            x[...] = 1
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

# read image from PATH
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path of image file')
args = parser.parse_args()
im0 = cv2.imread(args.image_path)

# convert from BGR to RGB, get original image dimensions
im0 = im0[:, :, ::-1].astype('float32')/255
height0 = im0.shape[0]
width0 = im0.shape[1]

# resize and normalise image
im = cv2.resize(im0, (256, 256))
input = img_transform(im).view(-1,3,256,256)

# run model
activation = nn.Sigmoid()
with torch.no_grad():
    mask = model(input.cuda())
    mask = activation(mask)

# Prepare data to be displayed
mask = np.array(mask[0].cpu())
mask = edit_label(mask)
mask = np.uint8(mask * 255)
thin = cv2.ximgproc.thinning(mask)
thin = cv2.resize(thin, (width0, height0))

# Probabilistic Line Transform
# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
# threshold: The minimum number of intersections to "*detect*" a line
# minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
# maxLineGap: The maximum gap between two points to be considered in the same line.
lines = cv2.HoughLinesP(thin, 1, np.pi/180, 0, 20, 0)
# Draw the lines
hough = np.zeros((height0, width0), dtype=np.uint8)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(hough, (x1, y1), (x2, y2), (255,255,255), 1)

# show results on screen
plt.figure(figsize=(15, 15))
title = ['Input Image', 'Resized Image', 'Predicted Mask', 'Thinned Mask', 'Hough Transform']
display_list = [im0, im, mask, thin, hough]
for i in range(len(display_list)):
    plt.subplot(3, 2, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
plt.show()
