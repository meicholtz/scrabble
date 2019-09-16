#! /usr/bin/env python
"""Test a 2D YOLO network (v2) on images."""
import argparse
import imghdr
import os

import ipdb
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import scipy.io

from models.keras_yolo import create_model, yolo_eval, yolo_head
import utils

parser = argparse.ArgumentParser(description="Test a 2D YOLO network (v2) on images.")
parser.add_argument(
    'model_path',
    help='path to h5 file containing trained YOLO network')
parser.add_argument(
    'data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'")
parser.add_argument(
    '-a', '--anchors_path',
    help='path to anchors file, defaults to model_data/yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))
parser.add_argument(
    '-c', '--classes_path',
    help='path to classes file, defaults to model_data/shape_classes.txt',
    default=os.path.join('model_data', 'shape_classes.txt'))
parser.add_argument(
    '-o', '--output_path',
    help='path to output test results (as a mat file), defaults to testing/temp.mat',
    default=os.path.join('testing', 'temp.mat'))
parser.add_argument(
    '-b', '--batch',
    help='batch size for testing images, defaults to 2',
    default=2)


def _main(args):
    # Parse input arguments
    model_path = os.path.expanduser(args.model_path)
    data_path = os.path.expanduser(args.data_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    output_path = os.path.expanduser(args.output_path)

    batch = args.batch

    assert model_path.endswith('.h5'), 'model_path must have .h5 extension'
    assert output_path.endswith('.mat'), 'output_path must have .mat extension'

    # Extract anchors and classes from input files
    anchors = utils.get_anchors(anchors_path)
    classes = utils.get_classes(classes_path)
    images, boxes = utils.get_data(data_path)

    # Create model and load weights from file
    model_body, model = create_model(images.shape[1:-1], int(boxes.shape[-1]), anchors, classes)
    model_body.load_weights(model_path)
    model.summary()




if __name__ == '__main__':
    _main(parser.parse_args())
