#!/usr/bin/env python

'''Utility functions for Scrabble project.'''

import numpy as np
import cv2
import os
from skimage.util import montage
# import ipdb

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
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1000, 1000)
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


def readlabels(file, ind='all'):
    '''Read labeled information (e.g. image filename, clicked corners) from file.

        Parameters
        ----------
        file : str
            Path to text file containing labeled Scrabble board information

        ind : int or 'all'
            Index of the file to read. Set ind = 'all' to read all available boards. [DEFAULT = 'all']
    '''
    if ind == "all":
        x = np.loadtxt(file, dtype=str)
        imgfile = os.path.join(home(), 'data', x[:, 0])  # full path to raw image
        pts = np.float32(x[:, 1:])
    else:
        x = np.loadtxt(file, dtype=str, skiprows=ind, max_rows=1)
        # imgfile = os.path.join(home(), 'data', x[:, 0])  # full path to raw image
        imgfile = os.path.join(home(), 'data', x[0])  # full path to raw image

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


def str2ind(imagefile, labelfile=os.path.join(home(), 'labels', 'labels.txt')):
    '''Convert filename string to numeric index from label file.'''
    x = np.loadtxt(labelfile, dtype=str)
    x = list(x[:, 0])

    try:
        ind = x.index(imagefile)
    except:
        print(imagefile, "not found in label file:", labelfile)
        ind = -1

    return ind


def display_board(squares):
    squares = squares.reshape((-1, 55, 55))
    m = montage(squares[:225], grid_shape=(15, 15))
    cv2.imshow("Montage", m)
    cv2.waitKey(0)


def file_from_str(strr):
    # which label file the strr is in

    # determine which index the string is in the labels file
    pass


def show_labels(img, textfile, pts = None):
    ''' Given inputs of an image and textfile (optional input: points to warp image) show text labels on top of the image.

        Parameters
        ----------
        img: numpy array
            input image. If the image is not warped, pass in pts.

        textfile: str
            full path to text file containing the labels for the image.

        pts: numpy array
            an array of 4 points containing the corners of the image
    '''
    # if points are given, warp the image
    if (pts != None):
        img = imwarp(img, pts)
    lf = open(textfile)
    for line in lf.readlines():
        if(line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[0] + 55)
        cv2.putText(img=img, org=(x,y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255))
    cv2.namedWindow("Text Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Text Overlay", 1000, 1000)
    cv2.imshow("Text Overlay", img)
    cv2.waitKey(0)

def count_letters(directory=os.path.join(home(), 'labels'), skip=['labels.txt', 'labels1.txt'], retboards=False):
    ''' Given inputs of an image and textfile (optional input: points to warp image) show text labels on top of the image.

        Parameters
        ----------
        directory: str
            directory containing label text files

        skip: list
            list of text files to skip

        retboards: Boolean
            if True return the total number of boards labeled

        :return a numpy array containing counts for each letter
    '''
    boards = 0
    letters = np.zeros(26)
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and filename not in skip:
            boards += 1
            f = open(os.path.join(directory, filename))
            for line in f.readlines():
                letter = line.split(' ')[0]
                if (letter == 'NONE' or ord(letter) < 65):
                    continue
                letters[ord(letter) - 65] += 1
        else:
            continue
    if(retboards):
        return letters, boards
    return letters
