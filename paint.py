import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import the model h5
model = load_model('digit_CNN_trained.h5')

# Global variables
run = False
ix, iy = -1, -1
follow = 25
img = np.zeros((512, 512, 1), dtype=np.uint8)  # Initial canvas

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

        # Preprocess the image for the CNN model
        gray = cv2.resize(img, (28, 28)) # Resize to 28x28
        gray = gray.astype('float32') / 255.0 # Normalize
        gray = np.expand_dims(gray, axis=0) # Add batch dimension
        gray = np.expand_dims(gray, axis=-1) # Add channel dimension

        # Prediction with the model)
        result = np.argmax(model.predict(gray), axis=1)[0]
        result_txt = f'pred : {result}'

        # Display the result on the canvas
        cv2.putText(img, result_txt, org=(25, follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255, 0, 0), thickness=1)
        follow += 25

    # Reset the canvas
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 1), dtype=np.uint8)
        follow = 25

# Setup the OpenCV window
cv2.namedWindow('Paint it')
cv2.setMouseCallback('Paint it', draw)

# Main loop to display the canvas
while True:    
    cv2.imshow("Paint it", img)
    if cv2.waitKey(1) == 27: # ESC key to quit
        break

cv2.destroyAllWindows()
