import numpy as np
import cv2

img = cv2.imread('C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\subaru.jpg', cv2.IMREAD_COLOR)
# img[55,550] = [255,255,255]
# px = img[55,55]

car_face = img[130:360, 330:750]
img[0:230,0:420] = car_face
print()
cv2.imshow('vroom vroom', img)
cv2.waitKey(0)
cv2.destroyAllWindows()