import cv2
import numpy as np

img=cv2.imread('nature1.jpg')

ret,threshold=cv2.threshold(img,12,205,cv2.THRESH_BINARY)

grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,threshold2=cv2.threshold(img,12,255,cv2.THRESH_BINARY)

cv2.imshow('Original',img)
cv2.imshow('Threshold',threshold)
cv2.imshow('Threshold2',threshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()


