import cv2
import numpy as np
import winsound


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))


while True:
    _, frame = cap.read()
    _, frame2 = cap.read()

    #checking for movements
    diff = cv2.absdiff(frame, frame2)

    #converting to gray
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    #changing to blur
    blur = cv2.GaussianBlur(gray,(5,5), 0)

    #threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    #dilation getting rid of specs
    dilated = cv2.dilate(thresh, None, iterations=3)

    #detection for moving or not
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #drawing the contours
    #cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    for i in contours:
        if cv2.contourArea(i) < 5000: #5000 is the threshold of detection
            continue
        x, y, w, h = cv2.boundingRect(i)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
        #winsound.Beep(500, 200) #fucking alarm

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()