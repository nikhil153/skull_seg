# Objective
- Segment skull from MR images using u-net models from brats2017 challenges

# Basic workflow
- Input: Nifti MR images and labels
- preproc (for computational feasibility) 
  - Use a dilated average skull segmentations (or other brain extraction methods) to remove most of the brain form the images
  - Generate patches 
- Segment skull using 3D patch-based U-Net style model 
- postproc
  - clean small disconnected erroneous data clusters using skimage package 

# Code organization 
- ./generate_patches.py: generate smaller patches from images for training and testing
- ./model.py: 3D Unet style model for segmenation 
- ./utils.py: helper funtions for reading, writing, and manupulating images for optional normalization 
- notebooks/UNet_Test.ipynb: driver code for training and testing images (starting point for playing around with this code) 
- notebooks/data_aug_stats: preprocessing code for creating and applying masks, as well as for checking distribution stats of augmented data
- notebooks/UNet_Test.py: python version of the UNet_Test.ipynb for running from cmd line (Not tested thoroughly). 
- notebooks/BET_test.ipynb: standalone notebook for run FSL BET pipeline for skull-segmentation. This is to be used as a baseline performance. This notebook needs to be run from a neurodocker. The instructions are listed in the notebook itself. 
