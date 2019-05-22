from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import cv2


# gather images that have been labelled
f = open('labels.txt')
dir = '/Users/Alex/Desktop/Summer 2019/scrabble/data/'
# since scrabble is 15 by 15 i should be divisible by 15
i = 825
# if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
s = int(i/15)
# data to be clustered
data = []
counter = 0
# number of boards to cluster
num_boards = 1
for line in f.readlines():
    strr = ''
    # split the line in the text file
    x = line.split()
    # store the image name
    img = dir + x[0]
    # read and resize the image
    img = cv2.imread(img, 0)
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
            square = square.reshape((-1))
            data.append(square)
    counter += 1
    if counter == num_boards:
        break

data = np.asarray(data)
n_samples, h, w = data.shape
n_features = data.shape[1]
n_classes = 30
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# this number im not sure where to get?
n_components = 100
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(data)

