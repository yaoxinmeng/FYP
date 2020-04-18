# Final Year Project
Edge Detection using Deep Learning for Indoor 3D Reconstruction
## Introduction
This project aims to determine the best deep learning model for the task of extracting structural edges from photographic images. This repository contains all the relevant resources used for this project as well as the findings from this project. A short explanation for each folder is given below.
## cleaned_data
This folder contains over 5000 photographic images of rooms sourced from the LSUN Scene Classification Challenge. These images have 
not been labelled and exists merely as a backup. Each image is 256 x 256 pixels.
## labelling
This folder contains all the necessary tools and resources for generating the dataset. 
###### resize.py
When starting from scratch, the first thing that needs to be done is to transfer all the images to a new local folder. In this code, each image is resized to 512 x 512 to make it easier to label, and each image is also labelled with a unique number. The new image is then saved to a local directory (also named cleaned_data for simplicity). To run the code, simply enter the following line in command. 
```
python resize.py ../cleaned_data --out_dir cleaned_data
```
###### label.py
This code generates a GUI for users to label their own images. Each input image is automatically taken from 
