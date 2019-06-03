#!/usr/bin/env python

'''Utility functions for Scrabble project.'''

import numpy as np
import cv2
import os
import pdb
from skimage.util import montage


def get_squares(file, num_boards, root=os.path.join(os.getcwd(), 'data')):
    """Extract individual squares (as a numpy array) from a select number of Scrabble boards.

        Parameters
        ----------
        file : str
            Path to text file containing labeled Scrabble boards

        num_boards : int
            Number of boards the user wishes to process

        root : str
            Path to root data directory. DEFAULT=data folder in top directory
    """

    f = open(file)
    squares = []
    counter = 0
    pixels = 825  # should be divisible by 15 because scrabble is 15x15
    s = int(pixels / 15)  # width and height of squares (in pixels)
    for line in f.readlines():
        # split the line in the text file
        x = line.split()
        imfile = os.path.join(root, x[0])

        # read and resize the image
        img = cv2.imread(imfile, 0)
        img = cv2.resize(img, (640, 480))
        pts = np.float32(eval("".join(x[1:])))
        img = imwarp(img, pts)

        # now we need to extract the tiles
        for j in range(15):
            for k in range(15):
                square = np.float32(dst[s * j: s + s * j, s * k: s + s * k])
                square = square.reshape((-1))
                squares.append(square)
        counter += 1
        if counter == num_boards:
            break
    squares = np.uint8(squares)
    return np.asarray(squares)


def imwarp(img, pts, sz=(825, 825)):
    '''Warp an image of a Scrabble board given an array of the board corners.

        Parameters
        ----------
        img : np.array
            Original image of Scrabble board

        pts : np.array
            4x2 array of points denoting the corners of the board in the image

        sz : tuple
            Desired size (in pixels) of output image. DEFAULT=(825,825)
    '''
    pts2 = np.float32([[0, 0], [sz[0], 0], [0, sz[1]], [sz[0], sz[1]]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    return cv2.warpPerspective(img, M, sz)


def squares_from_img(img):
    w, h = img.shape[0], img.shape[1]
    temp = []

    # since scrabble is 15 by 15 i should be divisible by 15
    i = w
    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i / 15)
    for j in range(15):
        for k in range(15):
            square = np.float32(img[s * j: s + s * j, s * k: s + s * k])
            temp.append(square)
    temp = np.uint8(temp)
    img = np.asarray(temp)
    return img


def get_board(file, index, squares=False):
    if(os.path.basename(os.getcwd()) != 'scrabble'):
        root = os.path.join(os.path.dirname(os.getcwd()), 'data/')
    else:
        root = os.path.join(os.getcwd(), 'data/')
    labelfile = file
    ind = index

    # Read data from labelfile
    x = np.loadtxt(labelfile, dtype=str, skiprows=ind, max_rows=1)
    imagefile = root + x[0]  # full path to raw image
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
    img2 = cv2.resize(img2, (825, 825))
    if(squares):
        temp = []
        # since scrabble is 15 by 15 pixels should be divisible by 15
        pixels = 825
        # if you divide pixels by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
        s = int(pixels / 15)
        for j in range(15):
            for k in range(15):
                square = np.float32(img2[s * j: s + s * j, s * k: s + s * k])
                temp.append(square)
        temp = np.uint8(temp)
        img2 = np.asarray(temp)
    return img2


def squares_to_board(squares):
    squares = squares.reshape((-1, 55, 55))
    m = montage(squares[:225], grid_shape=(15, 15))
    return m


def showboard(board):
    cv2.imshow("board", board)
    cv2.waitKey(0)


def display_board(squares):
    squares = squares.reshape((-1, 55, 55))
    m = montage(squares[:225], grid_shape=(15, 15))
    cv2.imshow("Montage", m)
    cv2.waitKey(0)
