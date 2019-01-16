import cv2
import numpy as np
import PIL

# def create_rec(p0, p1):
	
# 	P0, P1 = p0, p1
# 	sim_photo = np.zeros([1024, 768, 3])

# 	sim_photo[0:20, 0:20, 2] = 255

# 	return sim_photo
# photo_np = create_rec(0, 0)

b = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
g = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
r = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
# b = np.zeros((200, 300), dtype=np.uint8) + 255
# r = np.zeros((200, 300), dtype=np.uint8) + 255
# g = np.zeros((200, 300), dtype=np.uint8) + 255

x = np.random.randint(0, 480)
y = np.random.randint(0, 640)
hl, hs = 20, 15
wl, ws = 60, 45
h1, h2 = max(x - hl, 0), min(x + hl, 480)
w1, w2 = max(y - wl, 0), min(y + wl, 640)
h3, h4 = max(x - hs, 0), min(x + hs, 480)
w3, w4 = max(y - ws, 0), min(y + ws, 640)
r[h1:h2, w1:w2] = 255 
b[h1:h2, w1:w2] = 0
g[h1:h2, w1:w2] = 0

b[h3:h4, w3:w4] = 255
g[h3:h4, w3:w4] = 255

img = cv2.merge([b, g, r])

# photo_cv = cv2.fromar(photo_np)
# img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# the order of color in cv is [b,g,r]
# draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
# draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
# draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
draw_img3 = cv2.drawContours(img.copy(), contours, -1, (255, 255, 0), 3)

print("contours:",type(contours))
print("numbers",len(contours))

cv2.imshow("img", img)
# cv2.imshow("draw_img0", draw_img0)
# cv2.imshow("draw_img1", draw_img1)
# cv2.imshow("draw_img2", draw_img2)
cv2.imshow("draw_img3", draw_img3)

k = cv2.waitKey(0)
if k == 27:
	cv2.destroyAllWindows()

