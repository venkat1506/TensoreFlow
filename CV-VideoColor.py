import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red=np.array([60,0,0])
    upper_red=np.array([255,255,255])

    mask=cv2.inRange(hsv, lower_red, upper_red)
    res=cv2.bitwise_and(frame,frame, mask = mask)
    blur=cv2.GaussianBlur(res,(15,15),0)

    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('blur', blur)

    k=cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
