import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import pdb
import re

''' K means clustering with 30 classes: 26 letters, 1 blank time, 1 double letter, 1 triple letter, 1 empty'''


# gather images that have been labelled
f = open('labels.txt')
dir = '/Users/Alex/Desktop/Summer 2019/scrabble/data/'
# since scrabble is 15 by 15 i should be divisible by 15
i = 825
# if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
s = int(i/15)
# data to be clustered
data = []
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
    # now we need to extract the tiles
    for j in range(15):
        for k in range(15):
            fname = str(j) + str(k) + ".txt"
            square = np.float32(dst[s * j: s + s * j, s * k: s + s * k])
            square = square.reshape((9075))
            data.append(square)
    data = np.asarray(data)