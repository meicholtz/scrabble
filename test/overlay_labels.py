import cv2
import numpy as np
from utils import *
import os
import ipdb


img, pts = readlabels(os.path.join(home(), 'labels', 'labels.txt'), ind=12)
img = cv2.imread(img)
img = imwarp(img, pts)
lf = "/Users/Alex/Desktop/Summer-2019/scrabble/labels/Photo_2006-07-01_021.txt"
show_labels(img, lf)