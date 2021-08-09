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

The project was worked on in Python 3.9.1, and the model building was coded on [Jupyter notebook](https://jupyter.org/). Libraries used for the project can be found listed below, and a copy of their respective versions can be found in the [requirements.txt](https://github.com/oypjoshua/Data_Science/blob/main/Mask%20Detection/requirements.txt) file.

- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)

### Model Building and Tuning

From the various choices of model selection, [DenseNet201](https://www.mathworks.com/help/deeplearning/ref/densenet201.html) was selected for the model building. As can be seen in the [model building notebook](https://github.com/oypjoshua/Data_Science/blob/main/Mask%20Detection/face_mask.ipynb), an accuracy of >99% was achieved within the first 6 epochs. As such, I shall spare readers a fancy accuracy-loss graph. To ensure that the model was not overfitting, we evaluated the model on the test set, and was very contend with the 99.5% accuracy that was achieved. A [summary](https://github.com/oypjoshua/Data_Science/blob/main/Mask%20Detection/base_model_summary.ipynb) of DenseNet201 is available, and the model itself can be found [here](https://github.com/oypjoshua/Data_Science/blob/main/Mask%20Detection/mask_detection_128x128.rar).

### Real Time Analysis

For the live streaming, we used OpenCV's inhouse video capture to access the webcam. Upon receiving the image, we run it through OpenCV's [dnn.blobFromImage](https://docs.opencv.org/4.5.2/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7), using weights from [ImageNet](https://www.image-net.org/) to detect faces that may or may not be wearing masks. We then crop these images, process them, and run them through our pretrained model, to figure out if these faces have a mask on. Finally, we feed this infomation back to the live capture, tag it, and then show it.

## Difficulties

  - Selection of dataset (eventually settling on data set from kaggle), options included individually masking of faces with [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)
  - Selection of model (running multiple models and evetually deciding on DenseNet201) (Selection from mobilenet/ sequential/ resnet)
  - Selection of real time analysis face detection (haar cascades, mtcnn, local binary patterns, [blobFromImage](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/) 

## Project Extensions

  - Run multiface detection
  - Check for poorly worn mask
