#Aim:To create hand gesture recognising program using opencv lib
#note: venv stands for virtual environment to import and use all the python libraries. We have to seperately install all the required libraries through terminal command of vscode
#first we write code to turn on the camera 
#second from cvzone lib we import hand tracking module to detect hand (predefined)
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300

folder = "Data/A"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]  # We only use 1 hand
        x, y, w, h = hand['bbox']  # Creating the bounding box to crop the hands

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Print the dimensions of imgcrop before resizing
        print("Before resizing - imgcrop shape:", imgcrop.shape)

        imgcropshape = imgcrop.shape

        aspectratio = h / w
        if aspectratio > 1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))

            # Print the dimensions of imgresize after resizing
            print("After resizing - imgresize shape:", imgresize.shape)

            imgresizeshape = imgresize.shape
            wgap = math.ceil((300 - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (hcal, imgsize))

            # Print the dimensions of imgresize after resizing
            print("After resizing - imgresize shape:", imgresize.shape)

            imgresizeshape = imgresize.shape
            hgap = math.ceil((300 - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize

        cv2.imshow("Imagecrop", imgcrop)
        cv2.imshow("Imagewhite", imgwhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
