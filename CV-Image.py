import cv2

# Load an color image in grayscale
img = cv2.imread('nature.jpg',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('NAT.png',img)