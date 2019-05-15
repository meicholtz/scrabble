'''Double click to place a point. Place four points, one at each corner of the Scrabble board. The order of the points placed matters. The order is as follows: top left, top right, bottom left, bottom right.'''

import cv2
import numpy as np
import pdb
import argparse
import glob
import os


# set up argparse
parser = argparse.ArgumentParser(description='Double click to place a point. Place four points one at each corner of the Scrabble board. The order of the points placed matters. The order is as follows: top left, top right, bottom left, bottom right. Press ESC to skip an image. Press Q to quit.')
parser.add_argument('-d', '--directory', type=str, help='The directory containing images you want to label.', default=os.path.join(os.getcwd(), 'data'))
parser.add_argument('-f', '--file', help='The output file to write labels.', type=str, default=os.path.join(os.getcwd(), 'labels.txt'))
parser.add_argument('-r', '--reverse', help='label images in reverse order', action="store_true")
args = parser.parse_args()


class FourPoints:
    # this class is used to store four points that make up the corners of a Scrabble board
    def __init__(self):
        self.points = []
        self.img = ''

    # add a circle to the image based on where the mouse double clicks
    def add_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # x, y pos of mouse, 2 is the radius of the circle, the rest of the parameters are color
            cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
            self.points.append([x, y])

    def get_points(self):
        return self.points

    def new_points(self):
        self.points = []

    def set_img(self, i):
        self.img = i


def main(args):
    # from argparser
    dir = os.path.expanduser(args.directory)
    file = os.path.expanduser(args.file)

    # check to see if directory exists
    if not os.path.isdir(dir):
        raise Exception('Directory {} does not exist'.format(dir))

    # the a+ indicates that if the file does not exist, create it and also this file will be appended to
    f = open(file, "a+")
    # seek starts the reading the file from the beginning
    f.seek(0)
    # labelled is all of the names of images in the text file
    labelled = [x.split(' ')[0] for x in f.readlines()]
    # images is a list of all images in the directory that are .jpg
    images = glob.glob(dir + "/*.jpg")

    if args.reverse:
        images.reverse()
    # instantiate the four points class
    fourpoints = FourPoints()

    # for each image in the directory:
    for i in images:
        # check to see if image has been labelled
        if not os.path.basename(i) in labelled:
            img = cv2.imread(i, 0)
            # resize the image to fit for the window
            img = cv2.resize(img, (640, 480))
            cv2.namedWindow('image')
            fourpoints.set_img(img)
            # this tracks the mouse position
            cv2.setMouseCallback('image', fourpoints.add_point)
            cont = True
            while (cont):
                cv2.imshow('image', img)
                # if esc is hit, skip this image
                if cv2.waitKey(20) & 0xFF == 27:
                    cont = False
                    fourpoints.new_points()
                # if q is hit, close the file and exit the code
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    f.close()
                    exit()

                # if four points are placed stop waiting for points
                if len(fourpoints.get_points()) == 4:
                    # add the four points and file name to text file
                    f.write(os.path.basename(i) + " " + str(fourpoints.get_points()) + '\n')
                    # flush saves the file
                    f.flush()
                    # reset points
                    fourpoints.new_points()
                    cont = False

    # # show the image
    # img = cv2.imread("/Users/Alex/Desktop/Summer 2019/scrabble/data/Photo_2006-03-10_002.jpg", 0)
    # wd, ht = img.shape
    # # resize the image to fit for the window
    # img = cv2.resize(img, (640, 480))
    # cv2.namedWindow('image')
    # # this tracks the mouse position
    # cv2.setMouseCallback('image', fourpoints.add_point)
    #
    # while(1):
    #     cv2.imshow('image',img)
    #     # if esc is hit, exit
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break
    #     # if four points are placed stop waiting for points
    #     if len(fourpoints.get_points()) == 4:
    #         break

    # print(fourpoints.get_points())
    # strr = str(fourpoints.get_points())
    # f.write()
    #
    # i is the width and the height of the resulting image
    # since scrabble is 15 by 15 i should be divisible by 15
    i = 825
    pts1 = np.float32(fourpoints.get_points())
    pts2 = np.float32([[0, 0], [i, 0], [0, i], [i, i]])

    # if you divide i by 15 (number of rows and columns in Scrabble) you get the width and height (pixels) of each square
    s = int(i / 15)
    # M is the perspective matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst is the resulting flat image
    dst = cv2.warpPerspective(img, M, (i, i))
    cv2.destroyAllWindows()
    j, k = 0, 0
    cv2.imshow('output', dst[s * j: s + s * j, s * k: s + s * k])
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
