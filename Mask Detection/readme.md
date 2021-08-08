<h1 align="center">Face Mask Detector</h1>
<h4 align="center">oypjoshua</h2>


## Motivation

In 2019, Covid-19 has taken the world by storm. In the ensuing months, many countries have started implementations of mask regulations. Malls, restaurants, supermarkets, and other services around the globe have also started enforcing that all patrons are regulated to wear a mask for the safety of 

## Data

- [Kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) - 12k images from [Ashish Jangra](https://www.kaggle.com/ashishjangra27) (kaggle)

## Libraries (Requirements)

- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)

## Project Overview and Outline

### Introduction

  - Import packages
  - Visualisation and manipulation of data

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
