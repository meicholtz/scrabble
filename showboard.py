#!/usr/bin/env python

'''Show sample warped board from labeled images.'''

import argparse
import os
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='Show a sample Scrabble board from a labeled image, using image warping.')
parser.add_argument('labelfile', help='output text file from labeler')
parser.add_argument('index', help='index of the sample board to show')
parser.add_argument('-d', '--directory', type=str, help='directory containing images', default=os.path.join(os.getcwd(), 'data'))


def main(args):
    # Parse input arguments
    root = os.path.expanduser(args.directory)
    labelfile = os.path.expanduser(args.labelfile)
    ind = int(args.index)

    # Read data from labelfile
    x = np.loadtxt(labelfile, dtype=str, skiprows=ind, max_rows=1)
    imagefile = os.path.join(root, x[0])  # full path to raw image
    pts = eval(''.join(x[1:]))  # corners of the board

    # Read image
    img = cv2.imread(imagefile)
    img = cv2.resize(img, (640, 480))

    # Warp image
    sz = 15 * 32  # width and height of warped image (must be divisible by 15 since the board is 15x15)
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [sz, 0], [0, sz], [sz, sz]])
    M = cv2.getPerspectiveTransform(pts1, pts2)  # perspective matrix
    img2 = cv2.warpPerspective(img, M, (sz, sz))  # new image
    cv2.imshow(imagefile, img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main(parser.parse_args())
