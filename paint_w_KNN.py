"""
This script allows you to draw on a canvas using OpenCV and make predictions with a pre-trained KNN model.
The canvas is displayed using OpenCV, and you can draw digits on it using the mouse.
The script uses a pre-trained KNN model to predict the digit drawn on the canvas in real-time.

Usage:
    - Draw digits on the canvas using the mouse (right click to erase).
    - The script will predict the digit drawn using the pre-trained KNN model.
    - The predicted digit will be displayed on the canvas.
    - Press any key to exit the loop and close the canvas.
"""

import cv2
import numpy as np
import joblib 

# Import the model pkl (KNN)
pipeline = joblib.load('digit_KNN_trained.pkl')

# Global variables
run = False
ix, iy = -1, -1
follow = 25
img = np.zeros((512, 512, 1), dtype=np.uint8) # Initial canvas

# Function to draw and make a prediction
def draw(event, x, y, flag, params):
    global run, ix, iy, img, follow

    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y

    # Draw on the canvas
    elif event == cv2.EVENT_MOUSEMOVE and run:
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)

    # Prediction with the model and display the result on the canvas
    elif event == cv2.EVENT_LBUTTONUP:
        run = False

        # Preprocess the image for the KNN model
        gray = cv2.resize(img, (28, 28))    # Resize to 28x28
        gray = gray.flatten()               # Flatten the image to a 1D array

        # Prediction with the model
        result = pipeline.predict([gray])[0]
        result_txt = f'pred : {result}'

        # Display the result on the canvas
        cv2.putText(img, result_txt, org=(25, follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255, 0, 0), thickness=1)
        follow += 25

    # Reset the canvas
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 1), dtype=np.uint8)
        follow = 25

# Set up the canvas and the mouse event
cv2.namedWindow('Paint it')
cv2.setMouseCallback('Paint it', draw)

# Main loop to display the canvas
while True:    
    cv2.imshow("Paint it", img)

    # Press any key to exit
    if cv2.waitKey(1) != -1: 
        break

cv2.destroyAllWindows()
