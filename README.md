# Final Year Project
Edge Detection using Deep Learning for Indoor 3D Reconstruction
## Introduction
This project aims to determine the best deep learning model for the task of extracting structural edges from photographic images. This repository contains all the relevant resources used for this project as well as the findings from this project. A short explanation for each folder is given below.
## cleaned_data
This folder contains over 5000 photographic images of rooms sourced from the LSUN Scene Classification Challenge. These images have 
not been labelled and exists merely as a backup. Each image is 256 x 256 pixels.
## labelling
This folder contains all the necessary tools and resources for labelling the data. 
###### resize.py
When starting from scratch, the first thing that needs to be done is to resize all the images in the cleaned_data folder and transfer them to a new local folder. 
