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
    # split the line in the text file
    x = line.split()
    # store the image name
    img = dir + x[0]
    # read and resize the image
    img = cv2.imread(img)
    img = cv2.resize(img, (640, 480))
    # store the 4 points in x
    x = x[1:]
    # convert the points to a string
    pts1 = strr.join(x)
    # eval converts the string to an array
    pts1 = np.float32(eval(pts1))
    # pts1 are the corners and pts2 is the width and height
    pts2 = np.float32([[0, 0], [i, 0], [0, i], [i, i]])
    # M is the perspective matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst is the resulting flat image
    dst = cv2.warpPerspective(img, M, (i, i))
    cv2.imshow("finished", dst)
    cv2.waitKey(0)

