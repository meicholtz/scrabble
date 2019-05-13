import cv2
import numpy as np
import pdb


class FourPoints:
    def __init__(self):
        self.points = []

    def add_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            self.points.append([x,y])

    def get_points(self):
        return self.points


fourpoints = FourPoints()

# show the image
img = cv2.imread("/Users/Alex/Desktop/Summer 2019/scrabble/data/Photo_2006-03-10_002.jpg", 0)
wd, ht = img.shape
cv2.namedWindow('image')
cv2.setMouseCallback('image', fourpoints.add_point)

while(1):
    # hit escape to exit
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    if len(fourpoints.get_points()) == 4:
        break

print(fourpoints.get_points())

i = 1000
pts1 = np.float32(fourpoints.get_points())
pts2 = np.float32([[0,0],[i,0],[0,i],[i,i]])
M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img, M, (i, i))
cv2.destroyAllWindows()
cv2.imshow('output', dst)
cv2.waitKey(0)