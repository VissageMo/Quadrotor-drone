import cv2
import numpy as np


b = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
g = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
r = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

img = cv2.merge([b, g, r])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyWindow('test')
