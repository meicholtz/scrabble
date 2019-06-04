#!/usr/bin/env python

'''Show sample warped board from labeled images.'''

import argparse
import os
from utils import *

parser = argparse.ArgumentParser(description='Show a sample Scrabble board from a labeled image, using image warping.')
parser.add_argument('labelfile', help='output text file from labeler')
parser.add_argument('index', help='index of the sample board to show')


def main(args):
    # Parse input arguments
    labelfile = os.path.expanduser(args.labelfile)
    ind = int(args.index)

    # Read data from labelfile
    imgfile, pts = readlabels(labelfile, ind)

    # Process image
    img = improcess(imgfile, pts)

    # Show results
    imshow(img, imgfile)


if __name__ == '__main__':
    main(parser.parse_args())
