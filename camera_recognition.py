"""
This script captures video from the camera, processes the frames to recognize digits using a pre-trained CNN model, and displays the results in real-time.

Usage:
    - The script captures video from the default camera (index 0).
    - It processes each frame to detect and recognize digits.
    - The recognized digit is displayed on the video feed.
    - Press any key to exit the loop and close the video feed.
    
PS: preferably use a white sheet with handwritten digits
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import the model h5
model = load_model('digit_CNN_trained.h5')

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Main loop to capture video and recognize digits in real-time until any key is pressed
while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # Crop the image to the region of interest (ROI), convert to grayscale, and apply binary threshold
    img = img[200:400, 200:400]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    _, gray_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # Apply binary threshold
    

    # Preprocess the image for the CNN model
    gray = cv2.resize(gray_img, (28, 28))       # Resize to 28x28
    gray = gray.astype('float32') / 255.0       # Normalize
    gray = np.expand_dims(gray, axis=0)         # Add batch dimension
    gray = np.expand_dims(gray, axis=-1)        # Add channel dimension

    # Prediction with the model
    result = np.argmax(model.predict(gray), axis=1)[0]

    # Display the result on the canvas
    result = 'pred : {}'.format(result)
    cv2.putText(img, org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255, 0, 0), thickness=1)
    cv2.imshow("Life in grey", gray_img)
    cv2.imshow("Digits recognition in Real Time", img)
   
    # Press any key to exit
    if cv2.waitKey(1) != -1:
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
