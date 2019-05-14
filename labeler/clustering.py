import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import pdb
import re

# gather images that have been labelled
f = open('labels.txt')

for line in f.readlines():
    strr = ''
    x = line.split()
    x = x[1:]
    pts = strr.join(x)
    pts = eval(pts)
