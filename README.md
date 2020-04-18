# Final Year Project
Edge Detection using Deep Learning for Indoor 3D Reconstruction
## Introduction
This project aims to determine the best deep learning model for the task of extracting structural edges from photographic images. This repository contains all the relevant resources used for this project as well as the findings from this project. A short explanation for each folder is given below.
## cleaned_data
This folder contains over 5000 photographic images of rooms sourced from the LSUN Scene Classification Challenge. These images are from categories “bedroom”, “classroom”, “conference room” and “living room” and have not been labelled yet. Each image is 256 x 256 pixels and serves as a backup.
## labelling
This folder contains all the necessary tools and resources for generating the dataset. 
###### resize.py
When starting from scratch, the first thing that needs to be done is to transfer all the images to a new local folder. In this code, each image is resized to 512 x 512 to make it easier to label, and each image is also labelled with a unique number. The new image is then saved to a local directory (also named cleaned_data for simplicity). To run this code, simply enter the following line in command. 
```
python resize.py ../cleaned_data --out_dir cleaned_data
```
###### label.py
This code generates a GUI for users to label their own images. Each input image is automatically retrieved from the cleaned_data folder, and when the user is done, the code generates three outputs:
- Input image that is resized back to 256 x 256
- Output "mask" that contains all the labelled lines
- Backup copy of the input image (512 x 512)

Note that the original image in the cleaned_data folder is automatically deleted once the labelling is done, hence the need for a backup copy. These outputs are stored in the labelled_data directory in the following three folders respectively:
- data
- label
- history

To run this code, simply enter the following line in command. 
```
python label.py
```
###### augmentation.py
This code augments the dataset and enlarges the dataset by 4 times. This is done by flipping horizontally, brightening, and a combination of both. The output of this code is a HDF5 file that containes the compressed dataset. To run this code, simply enter the following line in command. 
```
python augmentation.py labelled_data
```
## training
