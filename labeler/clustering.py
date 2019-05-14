import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import pdb
import re

# gather images that have been labelled
f = open('labels.txt')
dir = '/Users/Alex/Desktop/Summer 2019/scrabble/data/'
# since scrabble is 15 by 15 i should be divisible by 15
i = 825
# if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
s = int(i/15)
for line in f.readlines():
    strr = ''
    x = line.split()
    img = dir + x[0]
    img = cv2.imread(img)
    img = cv2.resize(img, (640, 480))
    x = x[1:]
    pts1 = strr.join(x)
    pts1 = np.float32(eval(pts1))
    pts2 = np.float32([[0, 0], [i, 0], [0, i], [i, i]])
    # M is the perspective matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst is the resulting flat image
    dst = cv2.warpPerspective(img, M, (i, i))
    cv2.imshow("finished", dst)
    cv2.waitKey(0)

