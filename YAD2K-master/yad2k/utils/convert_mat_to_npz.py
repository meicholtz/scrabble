#! /usr/bin/env python
"""Convert a MATLAB data file (.mat) containing images and bounding boxes to a numpy data file (.npz) for training YOLO."""
import argparse
import os

import ipdb
import numpy as np

from scipy.io import loadmat

parser = argparse.ArgumentParser(description="Convert a MATLAB data file (.mat) containing images and bounding boxes to a numpy data file (.npz) for training YOLO.")
parser.add_argument('input', help="path to MATLAB data file (.mat) containing images and bounding boxes")
parser.add_argument('output', help="path to output numpy data file (.npz)")


def _main(args):
    # Parse input arguments
    root = os.path.expanduser(args.input)
    savefile = os.path.expanduser(args.output)

    # Read MATLAB data from file
    print("Reading data from file: ", root)
    data = loadmat(root)
    images = data['images']
    if len(data['images'].shape) == 3:
        images = images[:, :, :, None]
    images = np.transpose(images, (3, 0, 1, 2))
    boxes = data['boxes'].squeeze(axis=1)

    # Save to numpy data file, if requested
    if savefile:
        print("Saving data to file: ", savefile)
        np.savez(savefile, images=images, boxes=boxes)
    else:
        print("No output file was provided. Data is not saved.")


if __name__ == '__main__':
    _main(parser.parse_args())
