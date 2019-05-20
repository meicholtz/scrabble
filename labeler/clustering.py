import numpy as np
import cv2
from imutils import build_montages
from imutils import paths
from sklearn.cluster import KMeans
import pdb
import sys
sys.path.append('../test')
from test import montage


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

features = np.asarray(data)
kmeans = KMeans(n_clusters=30, random_state=0, max_iter=500).fit(features)
inds = np.where(kmeans.labels_ == 13)
fs = np.uint8(features)
fs = fs.reshape((225,55,-1))
fs = np.vstack(fs)
pdb.set_trace()
cv2.imshow("test", fs)
cv2.waitKey(0)
# montages = build_montages(features[inds], (128, 196), (7, 3))
# for montage in montages:
# 	cv2.imshow("Montage", montage)
# 	cv2.waitKey(0)
# # for x in features[inds]:
# #     cv2.imshow("yes", np.uint8(x).reshape((55,-1)))
# #     cv2.waitKey(0)
# pdb.set_trace()t