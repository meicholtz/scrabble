#! /usr/bin/env python
"""Train a 2D YOLO network (v2) using custom data."""
import argparse
import os
import numpy as np

from yad2k.models.keras_yolo import create_model, train
import yad2k.utils as utils

parser = argparse.ArgumentParser(description="Train a 2D YOLO network (v2) using custom data.")
parser.add_argument(
    '-a', '--anchors_path',
    help='path to anchors file, defaults to model_data/yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))
parser.add_argument(
    '-c', '--classes_path',
    help='path to classes file, defaults to model_data/shape_classes.txt',
    default=os.path.join('model_data', 'shape_classes.txt'))
parser.add_argument(
    '-d', '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images', defaults to images/shapes.npz",
    default=os.path.join('images', 'shapes.npz'))
parser.add_argument(
    '-o', '--output_path',
    help='path to prefix of trained models, defaults to training/shapes',
    default=os.path.join('training', 'shapes'))


def _main(args):
    # Parse input arguments
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    data_path = os.path.expanduser(args.data_path)
    output_path = os.path.expanduser(args.output_path)

    # Extract anchors, classes, images, and boxes from input files
    anchors = utils.get_anchors(anchors_path)
    classes = utils.get_classes(classes_path)
    # images, boxes = utils.get_data(data_path)
    # Alexander is adding this starting here:
    data = np.load(data_path, allow_pickle=True)
    images, boxes = data['images'], data['boxes']
    max_box_shape = max([box.shape for box in boxes])
    for i, box in enumerate(boxes):
        boxes[i] = np.zeros(max_box_shape, dtype=boxes[0].dtype)  # array of zeros (for padding)
        boxes[i][:box.shape[0], :box.shape[1]] = box
    boxes = np.array(boxes)  # convert to ndarray (as opposed to a list of ndarrays)

    # Train YOLO model
    model_body, model = create_model(images.shape[1:-1], int(boxes.shape[-1]), anchors, classes)
    model.summary()
    train(model, classes, anchors, images, boxes, output_path=output_path)


if __name__ == '__main__':
    _main(parser.parse_args())
