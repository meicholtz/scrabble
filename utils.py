#!/usr/bin/env python

'''Utility functions for Scrabble project.'''

import numpy as np
import cv2
import os
from skimage.util import montage
import ipdb

TILES = 15


def get_board(file, ind):
    '''Get warped image of a user-specified Scrabble board via labeled data.

        Parameters
        ----------
        file : str
            Path to text file containing labeled Scrabble board information

        ind : int
            Index of the board the user wishes to process
    '''
    imgfile, pts = readlabels(file, ind)  # get data from label file
    img = cv2.imread(imgfile)  # read image from file
    return imwarp(img, pts)  # return warped image


def get_squares(file, ind):
    '''Extract individual squares (as a numpy array) from a specific Scrabble board.

        Parameters
        ----------
        file : str
            Path to text file containing labeled Scrabble board information

        ind : int
            Index of the board the user wishes to process
    '''

    f = open(file)
    squares = []
    counter = 0
    pixels = 825  # should be divisible by 15 because scrabble is 15x15
    s = int(pixels / TILES)  # width and height of squares (in pixels)
    for line in f.readlines():
        # split the line in the text file
        x = line.split()
        imgfile = os.path.join(home(), 'data', x[0])

        # Read, resize, and warp the image
        img = cv2.imread(imgfile, 0)
        img = cv2.resize(img, (640, 480))
        pts = np.float32(eval("".join(x[1:])))
        img = imwarp(img, pts)

        # Extract tiles
        for j in range(TILES):
            for k in range(TILES):
                square = np.float32(img[s * j: s + s * j, s * k: s + s * k])
                square = square.reshape((-1))
                squares.append(square)
        counter += 1
        if counter == num_boards:
            break
    squares = np.uint8(squares)
    return np.asarray(squares)


def home():
    '''Get full path to root directory of Scrabble project code.'''
    if os.path.basename(os.getcwd()) != 'scrabble':
        root = os.path.dirname(os.getcwd())
    else:
        root = os.getcwd()

    return root


def imshow(img, name="Scrabble Board"):
    '''Display image of Scrabble board and wait for user to press a key.

        Parameters
        ----------
        img : np.array
            Image to display

        name : str
            Name of figure [DEFAULT = "Scrabble Board"]
    '''
    cv2.imshow(name, img)
    cv2.waitKey(0)


def imwarp(img, pts, sz=(825, 825)):
    '''Warp an image of a Scrabble board given an array of the board corners.

        Parameters
        ----------
        img : np.array
            Original image of Scrabble board

        pts : np.array
            Array of normalized coordinates denoting the corners of the board in the image

        sz : int/float OR tuple
            Desired size (in pixels) of output image [DEFAULT = (825, 825)]
    '''
    if type(sz) != tuple:
        sz = (sz, sz)
    pts1 = np.float32(pts.reshape(-1, 2) * img.shape[:2][::-1])
    pts2 = np.float32([[0, 0], [sz[0], 0], [0, sz[1]], [sz[0], sz[1]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, sz)


def readlabels(file, ind):
    '''Read labeled information (e.g. image filename, clicked corners) from file.

        Parameters
        ----------
        file : str
            Path to text file containing labeled Scrabble board information

        ind : int or 'all'
            Index of the file to read. Set ind = 'all' to read all available boards
    '''
    # TODO: Add "all" option for ind!
    if ind == "all":
        x = np.loadtxt(file, dtype=str)
    else:
        x = np.loadtxt(file, dtype=str, skiprows=ind, max_rows=1)
    imgfile = os.path.join(home(), 'data', x[:, 0])  # full path to raw image
    pts = np.float32(x[1:])  # corners of the board

    return imgfile, pts


def squares_from_img(img):
    w, h = img.shape[0], img.shape[1]
    temp = []

    # since scrabble is 15 by 15 i should be divisible by 15
    i = w
    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i / TILES)
    for j in range(TILES):
        for k in range(TILES):
            square = np.float32(img[s * j: s + s * j, s * k: s + s * k])
            temp.append(square)
    temp = np.uint8(temp)
    img = np.asarray(temp)
    return img


def squares_to_board(squares):
    squares = squares.reshape((-1, 55, 55))
    m = montage(squares[:225], grid_shape=(15, 15))
    return m


def display_board(squares):
    squares = squares.reshape((-1, 55, 55))
    m = montage(squares[:225], grid_shape=(15, 15))
    cv2.imshow("Montage", m)
    cv2.waitKey(0)
