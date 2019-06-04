#!/usr/bin/env python

'''Label corners of a Scrabble board in an image.'''

import cv2
import argparse
import glob
import os
from utils import *

parser = argparse.ArgumentParser(description='Double click to place a point. Place four points, one at each corner of the Scrabble board, in the following order: top left, top right, bottom left, bottom right. Press ESC to skip an image. Press Q to quit.')
parser.add_argument('-d', '--directory', type=str, help='the input directory containing images to label', default=os.path.join(home(), 'data'))
parser.add_argument('-f', '--file', help='the output file to write labels', type=str, default=os.path.join(os.getcwd(), 'labels.txt'))
parser.add_argument('-r', '--reverse', help='label images in reverse order', action="store_true")
parser.add_argument('-g', '--grayscale', help='show images as grayscale', action="store_true")

IMAGE_SIZE = (640, 480)  # resize the image prior to labeling


def main(args):
    # Extract relevant input arguments
    root = os.path.expanduser(args.directory)
    if not os.path.isdir(root):
        raise Exception("Directory '{}' does not exist".format(root))
    file = os.path.expanduser(args.file)
    reverse = args.reverse
    colorflag = not args.grayscale
    instructions = [
        'Click the top-left corner',
        'Click the top-right corner',
        'Click the bottom-left corner',
        'Click the bottom-right corner']

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
            img = cv2.resize(img, IMAGE_SIZE)
            cv2.namedWindow('image')
            cv2.setWindowTitle('image', imagefile)

            # User interaction loop
            cv2.setMouseCallback('image', addpoint, [img, pts])
            while True:
                # Update image
                cv2.imshow('image', img)
                addinstructions(img, instructions[len(pts)])

                # Wait for user input
                if cv2.waitKey(20) & 0xFF == 27:  # ESC --> skip image
                    pts = []
                    break
                elif cv2.waitKey(20) & 0xFF == ord('q'):  # q --> quit program
                    f.close()
                    exit()
                elif len(pts) == 4:
                    # Append filename and four points to text file
                    print(os.path.basename(imagefile), end=" ", file=f)
                    pts = np.float32(pts) / IMAGE_SIZE  # normalize
                    pts = pts.flatten()
                    for i in range(len(pts) - 1):
                        print("{0:0.4f}".format(round(pts[i], 4)), end=" ", file=f)
                    print("{0:0.4f}".format(round(pts[-1], 4)), file=f)
                    f.flush()  # flush saves the file
                    pts = []
                    break


def addpoint(event, x, y, flags, param):
    '''Add a circle on the image where user clicked and append corresponding point to a list.'''
    if event == cv2.EVENT_LBUTTONDOWN:
        img, pts = param  # unpack input parameters
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # draw circle on image
        pts.append([x, y])  # append new data point


def addinstructions(img, txt):
    '''Add text on the image instructing the user what to do next.'''
    # Set text parameters
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    margin = 12

    # Get dimensions and coordinates
    (wid, hei) = cv2.getTextSize(txt, font, font_scale, thickness)[0]
    x0 = img.shape[1] // 2 - wid // 2
    y0 = img.shape[0] // 2 - hei // 2
    topleft = (x0 - margin, y0 + margin)
    bottomright = (x0 + wid + margin, y0 - hei - margin)

    # Draw rectangle and text
    cv2.rectangle(img, topleft, bottomright, (0, 0, 0), cv2.FILLED)
    cv2.putText(img, txt, (x0, y0), font, font_scale, (255, 255, 255), thickness)


if __name__ == '__main__':
    main(parser.parse_args())
