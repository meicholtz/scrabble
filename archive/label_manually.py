#!/usr/bin/env python

'''Label letter tiles in an image of a Scrabble board.'''

import cv2
import os
import numpy as np
import ipdb
import re
import argparse
from utils import *


parser = argparse.ArgumentParser(description='Label letter tiles in an image of a Scrabble board.')
parser.add_argument('-l', '--labelfile', type=str, help='the full path to a label text file (see labeler.py)', default=os.path.join(home(), 'labels', 'labels.txt'))
parser.add_argument('-u', '--user', type=int, help='name (full or partial) of user running this code', default='alexander')

BLANK_LABEL = 'NONE'  # string for tiles that do not contain a letter


def main(args):
    # Extract relevant input arguments
    file = os.path.expanduser(args.labelfile)
    user = validateuser(args.user)

    # Show instructions
    print("Current user:", args.user)
    print("Press the letter on the keyboard that represents the letter shown.")
    print("If no letter is show, press the spacebar to see the next tile.")
    print("Press ESC to exit.")

    # Start labeling boards
    counter = 1
    for ind in range(0, 10000):
        if(ind % 3 != user):
            continue
        imgname = utils.readlabels(file, ind)[0]
        imgname = os.path.basename(imgname)
        imgname = os.path.splitext(imgname)[0]
        txtfile = imgname + '.txt'
        txtfile = os.path.join(utils.home(), 'labels', txtfile)
        f = open(txtfile, "a+")
        if not os.stat(txtfile).st_size == 0:
            continue
        # get the image of the board
        img = utils.get_board(file, ind)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("board", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("board", 400, 400)
        cv2.imshow("board", img)
        w, h = img.shape[0], img.shape[1]
        # get the individual tiles from the image
        sqs = utils.squares_from_img(img)
        # reshape to 4 dimensions so that x and y position can be tracked
        sqs = sqs.reshape((15, 15, sqs.shape[1], sqs.shape[2]))
        swh = sqs.shape[3]
        sq_width_height = float(swh / w)
        # for each tile:
        for y in range(0, 15):
            for x in range(0, 15):
                center_x = x * swh
                center_x = float(center_x / w)
                center_y = y * swh
                center_y = float(center_y / h)
                title = "Labeling tile: {}, boards completed: {}".format(counter % 225, np.floor(counter / 255))
                cv2.imshow(title, sqs[y][x])
                c = cv2.waitKey(0)
                # if the user hits escape quit the code
                if c == 27:
                    exit()
                text = filter_ocr(chr(c & 255))
                label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width_height, sq_width_height)
                f.write(label)
                counter += 1
                cv2.destroyWindow(title)
        f.close()
        cv2.destroyAllWindows()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def manual_label(square):
    cv2.imshow("Press key on keyboard once to label", square)
    c = cv2.waitKey(0)
    # if the user hits escape quit the code
    if c == 27:
        exit()
    return filter_ocr(chr(c & 255))


def filter_ocr(text):
    if(text == '' or text == ' '):
        return BLANK_LABEL
    # grab the first character
    t = text[0]
    # remove non letter text
    regex = re.compile('[^a-zA-Z]')
    t = regex.sub('', t)
    if (t == ''):
        return BLANK_LABEL
    # capitalize letter
    t = t.capitalize()
    return t


def black_pixel_percentage(img):
    w = img.shape[1]
    return float(1 - cv2.countNonZero(img) / w**2)


if __name__ == '__main__':
    main(parser.parse_args())
