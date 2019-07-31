import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()

    laplation = cv2.Laplacian(frame,cv2.CV_64F)            #find edges in an image
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)   #you can detect the edges of an image in both horizontal and vertical directions.
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(frame,100,200)

    cv2.imshow('Original',frame)
    cv2.imshow('Laplation',laplation)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()