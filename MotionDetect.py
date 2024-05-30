import cv2
import os
from datetime import datetime
from SendMail import send_email
from deepface import DeepFace


# wait 10000ms ( 10ms )
wait_sec = 10000

# capture image
cap = cv2.VideoCapture(0)

# make substraction
fgbg = cv2.createBackgroundSubtractorMOG2()

# ATTENTION: here revised to check if image read succesfully
while True:

    # if press 'q' key, stop program
    if 0xFF == ord('q'):
        break

    isMove = False

    # ret: if frame read correctly, frame: store frame
    ret, frame = cap.read()

    # apply substraction
    fgmask = fgbg.apply(frame)

    # threshold of image
    th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    # to find shape of image(th)
    # CHAIN_APPROX_SIMPLE to compress to memory
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        

        # calc area of shape
        area = cv2.contourArea(contour)

        # if larger than 10000, draw rectangle to enhance shape
        if area > 10000:

            print("motion detected!")
            isMove = True
            
    
    # draw and wait for 1ms
    cv2.imshow('frame', frame)
    if isMove:
        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Define the path to save the image
        image_path = f"./archive/motion_{timestamp}.jpg"
        
        # Save the frame as an image
        cv2.imwrite(image_path, frame)
        send_email(image_path)
        face_analysis = DeepFace.analyze(img_path = image_path)
        print(face_analysis)

        cv2.waitKey(wait_sec)

    cv2.waitKey(1)



cap.release()
cv2.destroyAllWindows()