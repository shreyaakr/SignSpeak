import mediapipe as mp
import cv2
import numpy as np
import time
from PIL import Image

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Demo video settings
DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

# List for storing hand gesture data
my_list = []

# Function for resizing images
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # Resize based on width or height
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Initialize video capture from webcam
vid = cv2.VideoCapture(0)

# Get video properties
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(vid.get(cv2.CAP_PROP_FPS))

# Setup video output file
codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

# List of finger tip landmarks for hand gestures
finger_tips = [8, 12, 16, 20]
thumb_tip = 4


# Start video processing loop
while True:
    ret, img = vid.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand landmarks with MediaPipe
    results = hands.process(img)

    # Convert image back to BGR for OpenCV
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # List to track the fold status of fingers
            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)

                if lm_list[tip].x < lm_list[tip - 2].x:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Gesture detection based on landmarks
            

            if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[12].y:
                cv2.putText(img, "washroom", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("1")

            if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                cv2.putText(img, "Food", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("2")
                sameer = "two"

                   # three
            if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                cv2.putText(img, "three", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("3")
                sameer="three"

           

            # five
            if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                0].x < \
                    lm_list[5].x:
                cv2.putText(img, "Hi", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("5")
                sameer="Five"
                # six
            if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                0].x < \
                    lm_list[5].x:
                cv2.putText(img, "Water", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("6")
                sameer="Six"
           
           
            # NINE
            if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                0].x < \
                    lm_list[5].x:
                cv2.putText(img, "Im okay", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("9")
                sameer="Nine"
            # A
           
            # B
            if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x > lm_list[8].x:
                cv2.putText(img, "water", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("B")
                sameer="B"
            # c
            if lm_list[2].x < lm_list[4].x and lm_list[8].x > lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                    lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x:
                cv2.putText(img, "Im Okay", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("C")
            # d
           

            # E
            if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                0].x < \
                    lm_list[5].x and lm_list[4].y > lm_list[6].y:
                cv2.putText(img, "NO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                my_list.append("E")

            # Additional gesture conditions here...

            # Draw the hand landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    # Resize frame for display
    frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    frame = image_resize(image=frame, width=640)

    # Display frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Stop the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources and close all windows
vid.release()
out.release()
cv2.destroyAllWindows()
