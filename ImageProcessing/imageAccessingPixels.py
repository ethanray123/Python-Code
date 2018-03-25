import cv2
import numpy as np
import matplotlib.pyplot as plt

img  =  cv2.imread('C:\\Users\\My PC\\PycharmProjects\\ImageProcessing\\venv\\subaru.jpg', 0)
# cv2.IMREAD_GRAYSCALE
# IMREAD_GRAYSCALE = 0
# IMREAD_COLOR = 1
# IMREAD_UNCHANGED = -1

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

height, width = img.shape
print(height)
print(width)

#x => width
#y => height

for x in range(0, width):
    for y in range(0, height):
        # print(str(x)+" "+str(y))
        if img[y][x] >= 100:
            img[y][x] = 255
        else:
            img[y][x] = 0


#
# for x in range(0,10):
#     for y in range(0,10):
#         print(img[x][y])
cv2.imwrite('C:\\Users\\My PC\\PycharmProjects\\ImageProcessing\\venv\\cargray.png', img)
