#! /usr/bin/env python
"""Convert a directory of images (png) and bounding boxes (txt) to a numpy array (npz) for training YOLO.

This script was initially created to handle synthetic data generated via MATLAB (makeSyntheticImages2.m), but it may have utility elsewhere.
"""
import argparse
import cv2
import glob
import os

import ipdb
import numpy as np

parser = argparse.ArgumentParser(description="Convert a directory of images (png) and bounding boxes (txt) to a numpy array (npz) for training YOLO.")
parser.add_argument('input', help="path to input directory of images (png) and bounding boxes (txt)")


def _main(args):
    # Parse input arguments
    root = os.path.expanduser(args.input)
    if root[-1] == os.path.sep:
        root = root[:-1]
    savefile = root + '.npz'

    # Collect and sort png and txt files
    pngfiles = glob.glob(os.path.join(root, '*.png'))
    pngfiles.sort()
    txtfiles = glob.glob(os.path.join(root, '*.txt'))
    txtfiles.sort()

    # Initialize useful parameters
    num_images = len(pngfiles)
    im = cv2.imread(pngfiles[0])
    allimages = np.empty((num_images,) + im.shape[:-1], dtype=np.uint8)
    allboxes = []

    # Parse the list of images and bounding box files
    for ii in range(num_images):
        allimages[ii, :, :] = cv2.imread(pngfiles[ii], cv2.IMREAD_GRAYSCALE)

        boxes = []
        with open(txtfiles[ii]) as f:
            line = f.readline()
            cnt = 1
            while line:
                print("{}".format(line.strip()))
                boxes.append(line.strip().split(' '))
                line = f.readline()
                cnt += 1

        allboxes.append(np.array(boxes, dtype=np.uint))

    # Save numpy data to file, if requested
    if savefile:
        print("Saving data to file: ", savefile)
        np.savez(savefile, images=allimages, boxes=allboxes)
    else:
        print("No output file was provided. Data is not saved.")


if __name__ == '__main__':
    _main(parser.parse_args())
