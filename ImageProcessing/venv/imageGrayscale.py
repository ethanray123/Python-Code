import cv2
import numpy as np
import matplotlib.pyplot as plt

img  =  cv2.imread('C:\\Users\\My PC\\PycharmProjects\\ImageProcessing\\venv\\subaru.jpg', 0)
# cv2.IMREAD_GRAYSCALE
# IMREAD_GRAYSCALE = 0
# IMREAD_COLOR = 1
# IMREAD_UNCHANGED = -1

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.plot([50,100],[80,100], 'c', linewidth=5) # plot a line
# plt.imshow(img, cmap="gray", interpolation="bicubic")
# plt.show()

cv2.imwrite('C:\\Users\\My PC\\PycharmProjects\\ImageProcessing\\venv\\cargray.png',img)