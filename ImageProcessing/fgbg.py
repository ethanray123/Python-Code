import cv2
import numpy as np

img1 = cv2.imread('bg.png')
img2 = cv2.imread('e.png')

rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
# add = img1 + img2
# add = cv2.add(img2,img1)
# weighted = cv2.addWeighted(img1, 0.9, img2, 0.1, 0)

# cv2.imshow('weighted', weighted)
# cv2.imshow('mask',mask)

mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()