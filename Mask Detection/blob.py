import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import time
# facial recognition types:
# haar_cascade = cv.CascadeClassifier('haar_face.xml')
# detector = MTCNN()

# detector = MTCNN()
    # MTCNN:
    
    # boxes = detector.detect_faces(frame)
    # if boxes:
    #     box = boxes[0]['box']
    #     conf = boxes[0]['confidence']
    #     x, y, w, h = box[0], box[1], box[2], box[3]
    #     cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 1)

# haar_cascade = cv.CascadeClassifier('haar_face.xml')
    # HAAR CASCADE:

    # faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 1)
capture = cv.VideoCapture(0)
net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
frameHeight = 480
frameWidth = 640
conf_threshold = 0.9

while True:
    isTrue, frame = capture.read()
    width = int(frame.shape[1])
    height = int(frame.shape[0])

    blob = cv.dnn.blobFromImage(frame, 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv.imshow('Video', frame)
    	# loop over the detections
    if cv.waitKey(20) & 0xFF == ord('d'):
        cv.destroyAllWindows()
        # print(detections)
        print(frame.shape)
        break

capture.release()