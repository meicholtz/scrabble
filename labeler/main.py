import cv2
import numpy as np
import pdb
import glob
import argparse
import os

''' Double click to place a point. Place four points one at each corner of the Scrabble board. The order of the points 
placed matters. The order is as follows: top left, top right, bottom left, bottom right.'''

# this class is used to store four points that make up the corners of a Scrabble board
class FourPoints:
    def __init__(self):
        self.points = []

    # add a circle to the image based on where the mouse double clicks
    def add_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # x, y pos of mouse, 2 is the radius of the circle, the rest of the parameters are color
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            self.points.append([x,y])

    def get_points(self):
        return self.points

    def new_points(self):
        self.points = []


# set up argparse
parser = argparse.ArgumentParser(description='Double click to place a point. Place four points one at each corner of '
                                             'the Scrabble board. The order of the points placed matters. '
                                             'The order is as follows: top left, top right, '
                                             'bottom left, bottom right.')

parser.add_argument('-d', '--directory', type=str, help='The directory containing images you want to label.',
                    default=os.path.join(os.getcwd(), '/data'))
parser.add_argument('-o', '--order', type=str, help='The order in which you want to label images.',
                    choices=['top', 'bottom'], default='top')
parser.add_argument('-f', '--file', help='The output file to write labels.', type=str, default='labels.txt')
parser.print_help()
args = parser.parse_args()

def main(args):
    dir = args.directory
    order = args.order
    file = args.file

    fourpoints = FourPoints()


    # show the image
    img = cv2.imread("/Users/Alex/Desktop/Summer 2019/scrabble/data/Photo_2006-03-10_002.jpg", 0)
    wd, ht = img.shape
    # resize the image to fit for the window
    img = cv2.resize(img, (640, 480))
    cv2.namedWindow('image')
    # this tracks the mouse position
    cv2.setMouseCallback('image', fourpoints.add_point)

    while(1):
        cv2.imshow('image',img)
        # if esc is hit, exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
        # if four points are placed stop waiting for points
        if len(fourpoints.get_points()) == 4:
            break

    print(fourpoints.get_points())

    # i is the width and the height of the resulting image
    # since scrabble is 15 by 15 i should be divisible by 15
    i = 825
    pts1 = np.float32(fourpoints.get_points())
    pts2 = np.float32([[0,0], [i,0], [0,i], [i,i]])

    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i/15)
    # M is the perspective matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst is the resulting flat image
    dst = cv2.warpPerspective(img, M, (i, i))
    cv2.destroyAllWindows()
    j,k = 0,0
    cv2.imshow('output', dst[s*j : s + s*j, s * k : s + s*k])
    cv2.imshow('full', dst)
    cv2.waitKey(0)

    # j is the row
    # k is the column
    # s is the width and height in pixels of each square
    for j in range(15):
        for k in range(15):
            fname = str(j) + str(k) + ".txt"
            np.savetxt(fname, dst[s * j: s + s * j, s * k: s + s * k])
            pdb.set_trace()



if __name__ == '__main__':
    main(parser.parse_args())
