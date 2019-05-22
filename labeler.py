#!/usr/bin/env python

'''Label corners of a Scrabble board in an image.'''

import cv2
import numpy as np
import argparse
import glob
import os


parser = argparse.ArgumentParser(description='Double click to place a point. Place four points, one at each corner of the Scrabble board, in the following order: top left, top right, bottom left, bottom right. Press ESC to skip an image. Press Q to quit.')
parser.add_argument('-d', '--directory', type=str, help='the input directory containing images to label', default=os.path.join(os.getcwd(), 'data'))
parser.add_argument('-f', '--file', help='the output file to write labels', type=str, default=os.path.join(os.getcwd(), 'labels.txt'))
parser.add_argument('-r', '--reverse', help='label images in reverse order', action="store_true")
parser.add_argument('-g', '--grayscale', help='show images as grayscale', action="store_true")


def main(args):
    # Extract relevant input arguments
    root = os.path.expanduser(args.directory)
    if not os.path.isdir(root):
        raise Exception("Directory '{}' does not exist".format(root))
    file = os.path.expanduser(args.file)
    reverse = args.reverse
    colorflag = not args.grayscale

    # Get images to be labeled
    imagefiles = glob.glob(os.path.join(root, "*.jpg"))  # image filename list
    if reverse:
        imagefiles.reverse()
    if len(imagefiles) == 0:
        print("There are no images in the directory:", root)
        print("Exiting program.")
        exit()

    # Setup output file
    f = open(file, "a+")  # create file (if it does not exist) to append data
    f.seek(0)  # start from beginning of file
    labeled = [x.split(' ')[0] for x in f.readlines()]  # labeled image list

    # Parse images and label if not already in output list
    pts = []  # instantiate list of points
    for imagefile in imagefiles:
        if not os.path.basename(imagefile) in labeled:
            # Load image and setup window
            img = cv2.imread(imagefile, colorflag)
            img = cv2.resize(img, (640, 480))
            cv2.namedWindow('image')
            cv2.setWindowTitle('image', imagefile)

            # Wait for user to skip, quit, or click on corners
            cv2.setMouseCallback('image', addpoint, [img, pts])
            while True:
                cv2.imshow('image', img)
                if cv2.waitKey(20) & 0xFF == 27:  # ESC --> skip image
                    pts = []
                    break
                elif cv2.waitKey(20) & 0xFF == ord('q'):  # q --> quit program
                    f.close()
                    exit()
                elif len(pts) == 4:
                    # add the four points and file name to text file
                    f.write(os.path.basename(imagefile) + " " + str(pts) + '\n')
                    f.flush()  # flush saves the file
                    pts = []
                    break


def addpoint(event, x, y, flags, param):
    '''Add a circle on the image where user clicked and append corresponding point to a list.'''
    if event == cv2.EVENT_LBUTTONDOWN:
        img, pts = param  # unpack input parameters
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # draw circle on image
        pts.append([x, y])  # append new data point


if __name__ == '__main__':
    main(parser.parse_args())
