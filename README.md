# Pneumonia Detection

This is a project I worked on for the AI for Healthcare Nanodegree by Udacity. It's an open ended implementation of a model that detects pneumonia, complete with exploratory data analysis, data augmentation, train/test splits, training, inference, and an FDA plan.

See the starter [here](https://github.com/udacity/AIHCND_C2_Starter).

The data can be downloaded from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). It consists of 112,120 X-ray images with disease labels from 30,805 unique patients. Patient data is also included in a csv.

## Installation
1. Install Anaconda if you don't have it already
2. Run `conda env create -f environment.yml`
3. Then run `conda activate medical`
This installs python 3.8, numpy, pandas, scikit-learn, matplotlib, tensorflow-gpu and keras among other dependencies.

## Files
The main files are
- EDA.ipynb (exploratory data analysis)
- Build and train model.ipynb
- Inference.ipynb

dcm files are specialized medical files containing an image and metadata. They are used to test clinical workflow.

## Results
See FDA_Submission.md or the pdf.

I was able to identify 68% of the positive pneumonia cases in the dataset, but that comes with a lot of false positives due to low precision. However, 87% of the negative cases are correctly identified with very few false negatives. Thus, the value of the model lies in its detection of non-pneumonia images, which can help the radiologist focus on unknown cases.
