#!/usr/bin/env python

'''Sort Scrabble board labels by image filename.'''

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Sort Scrabble board labels by image filename.')
parser.add_argument('labelfile', help='output text file from labeler')


def main(args):
    # Parse input arguments
    file = os.path.expanduser(args.labelfile)

    # Read labeled data from file
    x = np.loadtxt(file, dtype=str)

    # Sort based on image filename (first column)
    y = x[x[:, 0].argsort()]
    if np.array_equal(x, y):
        print("The labeled data is already sorted. Exiting function.")
    else:
        np.savetxt(file, y, fmt="%s")


if __name__ == '__main__':
    main(parser.parse_args())
