# Pneumonia Detection

This is a project I worked on for the AI for Healthcare Nanodegree by Udacity. It's an end-to-end implementation of a model that detects pneumonia, complete with EDA, data augmentation, train/test splits, training, inference, and an FDA plan.

See the starter [here](https://github.com/udacity/AIHCND_C2_Starter).

The data can be downloaded from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). It consists of 112,120 X-ray images with disease labels from 30,805 unique patients. Patient data is also included in a csv.

## Installation
1. Install Anaconda if you don't have it already
2. Run `conda env create -f environment.yml`
3. Then run `conda activate medical`

## Files
The main files (with explanatory names) are
- EDA.ipynb
- Build and train model.ipynb
- Inference.ipynb

dcm files are specialized medical files containing an image and metadata. They are used to test clinical workflow.
