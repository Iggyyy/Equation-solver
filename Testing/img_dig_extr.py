import cv2
import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image

#imgPIL = Image.open(r"Testing\digs.png") 

#img = plt.imread(r"Testing\digs1.png")
img = cv2.imread(r"Testing\nums.png")
#print(img)

print(img.shape)
print(img.dtype)

#b,g,r = cv2.split(img)

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_tresh = np.array([0,0,0])
upper_tresh = np.array([180,255,30])

#mask = cv2.inRange(hsv, lower_tresh, upper_tresh)
#res = cv2.bitwise_and(img, img, mask=mask)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 127,255,0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)

#img = cv2.drawContours(img, contours, 2, (0,255,0), 3)
for i in range(1, len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

while(1):

    


    k = cv2.waitKey(5)
    if k == 13:
        break
    cv2.imshow('Img', img)