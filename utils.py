import numpy as np
import cv2
import pdb
from skimage.util import montage

def get_squares(file, num_boards):
    """ function that takes in a text file and number of boards and returns the flattened, individual squares
    in a numpy array.


            Parameters
            ----------
            f : str
                The pathname to a text file containing labeled Scrabble boards

            num_boards : int
                Number of boards the user wishes to process
                :rtype: object
            """
    # TODO: change directory to be universal
    directory = '/Users/Alex/Desktop/Summer-2019/scrabble/data/'
    f = open(file)
    squares = []
    counter = 0
    # since scrabble is 15 by 15 i should be divisible by 15
    i = 825
    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i / 15)
    for line in f.readlines():
        strr = ''
        # split the line in the text file
        x = line.split()
        img = directory + x[0]

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
                square = np.float32(dst[s * j: s + s * j, s * k: s + s * k])
                square = square.reshape((-1))
                squares.append(square)
        counter += 1
        if counter == num_boards:
            break
    squares = np.uint8(squares)
    return np.asarray(squares)

def squares_from_img(img):
    squares = []
    # since scrabble is 15 by 15 i should be divisible by 15
    i = 825
    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i / 15)
    for j in range(15):
        for k in range(15):
            square = np.float32(img[s * j: s + s * j, s * k: s + s * k])
            square = square.reshape((-1))
            squares.append(square)
    pdb.set_trace()
    squares = np.uint8(squares)
    return np.asarray(squares)


def get_board(file, index):
    root = '/Users/Alex/Desktop/Summer-2019/scrabble/data/'
    labelfile = file
    ind = index

    # Read data from labelfile
    x = np.loadtxt(labelfile, dtype=str, skiprows=ind, max_rows=1)
    imagefile = root + x[0] # full path to raw image
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
    return img2

def display_board(squares):
    m = montage(squares, grid_shape=(15,15))
    cv2.imshow("Montage", m)
    cv2.waitKey(0)
