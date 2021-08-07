import numpy as np
import cv2 as cv
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

model = tf.keras.models.load_model('mask_detection_128x128.h5', custom_objects = None, compile = True, options = None)

# imagedatagenerator/plt.imshow uses RGB
# OpenCV uses BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


## Preprocessing the detected image, to use in model which has in input shape of (128, 128, 3), colour in BGR
def preprocess_face(img):
    processed_img = cv.resize(img, (128,128), interpolation = cv.INTER_CUBIC)
    processed_img = np.array(processed_img/ 255)
    return(np.array([processed_img]))


## model to detect faces with or without masks. confidence threshold can be tuned 
net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
conf_threshold = 0.90

## height and width of live video enviornment(to be changed as per camera specification)
frameHeight = 480
frameWidth = 640

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    width = int(frame.shape[1])
    height = int(frame.shape[0])

    ## Detecting faces with or without mask using OpenCV's blobFromImage:
    blob = cv.dnn.blobFromImage(frame, 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if len(detections) != 0:

        ## Extracting faces from live video:
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                ## Taking the image and transforming it into RGB since our model processes RGB
                face_img = frame[y1:y2, x1:x2]
                face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

                ## model prediction for mask. model predicts if person is not wearing mask
                ## closer to 1 if not wearing mask, closer to 0 if wearing mask

                mask_percentage = round(1 - model.predict(preprocess_face(face_img))[0][0], 3) * 100
                if round(mask_percentage) >= 50:
                    cv.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)            
                    cv.putText(frame, 'Mask Present', (x1, y1 - 5), cv.FONT_HERSHEY_PLAIN, 0.75, GREEN, 1, cv.LINE_AA)
                if round(mask_percentage) <= 50:
                    cv.rectangle(frame, (x1, y1), (x2, y2), RED, 1)
                    cv.putText(frame, 'No Mask', (x1, y1 - 5), cv.FONT_HERSHEY_PLAIN, 0.75 , RED, 1, cv.LINE_AA)
        
    cv.imshow('Video', frame)

    ## Pressing esc key to terminate:
    if cv.waitKey(20) & 0xFF == 27:
        print(width, height)
        cv.destroyAllWindows()
        break

capture.release()

cv.destroyAllWindows()