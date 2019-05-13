import cv2
import numpy as np


class FourPoints:
    def __init__(self):
        self.points = []

    def add_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            self.points.append((x,y))

    def get_points(self):
        return self.points


fourpoints = FourPoints()

# show the image
img = cv2.imread("/Users/Alex/Desktop/Summer 2019/scrabble/data/Photo_2006-03-10_002.jpg", 0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', fourpoints.add_point)

while(1):
    # hit escape to exit
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

print(fourpoints.get_points())