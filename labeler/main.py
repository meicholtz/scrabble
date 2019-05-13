import cv2
import numpy as np
from pyimagesearch.transform import four_point_transform



def add_point(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 5, (255,0,0), -1)
        ix, iy = x,y


# show the image
img = cv2.imread("/Users/Alex/Desktop/Summer 2019/scrabble/data/Photo_2006-03-10_002.jpg", 0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', add_point)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break