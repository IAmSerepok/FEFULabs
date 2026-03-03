import numpy as np
import cv2 as cv


img1 = cv.imread('1.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('2.png')
img3 = cv.imread('3.png')

img1 = cv.resize(img1, (640, 480))
img2 = cv.resize(img2, (640, 480))

img2 = cv.GaussianBlur(img2, (5, 5), 0.0)
img2 = cv.bilateralFilter(img2, 9, 75, 75)

img1 = cv.copyMakeBorder(img1, 3, 3, 3, 3, cv.BORDER_CONSTANT)

img1 = cv.rotate(img1, 90)

img3 = cv.addWeighted(img2, 0.5, img2, 0.5, 0.5)

cv.imshow('test1', img1)
cv.imshow('test2', img2)
cv.imshow('test3', img3)

print(img1.shape)

cv.waitKey(0)
cv.destroyAllWindows()
