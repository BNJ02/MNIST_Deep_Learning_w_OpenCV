import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import the model h5
model = load_model('digit_trained.h5')

##### open cv for capture and predicting through camera #####
##### cv2

# loop until enter is pressed
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
    img = img[200:400, 200:400]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("gray_wind", gray)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.reshape(1, 784)

    results = model.predict(gray)
    print(results)

    index = np.argmax(results)
    result = 'cnn : {}'.format(index)
    cv2.putText(img, org=(25,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
    # prob = 'prob : {}%'.format(results[0][index]*100)
    # cv2.putText(img, org=(25,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= prob, color=(255,0,0), thickness=1)
    cv2.imshow("image", img)
   
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()