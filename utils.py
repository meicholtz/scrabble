import numpy as np
import cv2



def get_squares(file, num_boards):
    """ function that takes in a text file and number of boards and returns the flattened, individual squares
    in a numpy array.


            Parameters
            ----------
            f : str
                The pathname to a text file containing labeled Scrabble boards

            num_boards : int
                Number of boards the user wishes to process
            """
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
        # read and resize the image
        img = cv2.imread(img, cv2.CV_8UC1)
        img = cv2.resize(img, (640, 480))
        img = cv2.medianBlur(img, 1)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
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
    return np.asarray(squares)
