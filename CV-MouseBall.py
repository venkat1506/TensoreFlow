# import cv2
# events = [i for i in dir(cv2)
#              if 'EVENT' in i]
# print(events)


import cv2
import numpy as np
img = np.ones((512,450,4), np.uint8)

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),80,(355,0,0),-1)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break


cv2.destroyAllWindows()
