<h1 align="center">Face Mask Detector</h1>
<h4 align="center">oypjoshua</h2>


## Motivation

In 2019, Covid-19 has taken the world by storm. In the ensuing months, many countries have started implementations of mask regulations. Malls, restaurants, supermarkets, and other services around the globe have also started enforcing that all patrons are regulated to wear a mask for the safety of patrons and others. In a bid to automate such checks, an efficient and accurate live face mask detector is desired.

## Project Overview and Outline

### Introduction

### Data

For this project, we will be using a dataset of ~12,000 images from [Kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset), courtesy of [Ashish Jangra](https://www.kaggle.com/ashishjangra27) (kaggle). The images with the face mask were scrapped from google search, while those without are processed from the [CelebFace dataset](https://www.kaggle.com/jessicali9530/celeba-dataset), courtesy of [Jessica Li](https://www.kaggle.com/jessicali9530). The final dataset comprises the following:

  - Training set (10,000 images total)
    - With Mask (5,000 images)
    - Without Mask (5,000 images)
  - Validation set (800 images total)
    - With Mask (400 images)
    - Without Mask (400 images)
  - Test set (988 images total)
    - With Mask (479 images)
    - Without Mask (509 images)
 
### Libraries (Requirements)

We used the following libraries

- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)

### Model Building and Tuning

  - Model selection [DenseNet201](https://www.mathworks.com/help/deeplearning/ref/densenet201.html)
  - Model building
  - Results

### Real Time Analysis

  - Video streaming (OpenCV)
  - Face detection (With/ Without Mask)
  - Preprocessing face to fit model (128x128)
  - Model running on face

## Difficulties

  - Selection of dataset (eventually settling on data set from kaggle), options included individually masking of faces with [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)
  - Selection of model (running multiple models and evetually deciding on DenseNet201) (Selection from mobilenet/ sequential/ resnet)
  - Selection of real time analysis face detection (haar cascades, mtcnn, local binary patterns, [blobFromImage](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)

## Project Extensions

  - Run multiface detection
  - Check for poorly worn mask
