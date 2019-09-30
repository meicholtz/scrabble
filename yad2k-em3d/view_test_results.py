import sys
sys.path.insert(1, 'scrabble')
import argparse
import os
import scipy.io
import ipdb
import cv2
import utils

parser = argparse.ArgumentParser(description="Train a 2D YOLO network (v2) using custom data.")
parser.add_argument(
    '-a', '--anchors_path',
    help='path to anchors file, defaults to model_data/yolo_anchors.txt',
    default=os.path.join('model_data', 'scrabble_anchors.txt'))
parser.add_argument(
    '-c', '--classes_path',
    help='path to classes file, defaults to model_data/shape_classes.txt',
    default=os.path.join('model_data', 'scrabble_classes.txt'))
parser.add_argument(
    '-d', '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images', "
         "defaults to images/shapes.npz",
    default=os.path.join('model_data', 'scrabble_dataset.npz'))
parser.add_argument(
    '-t', '--test_results',
    help='path to a .mat file of test results',
    default=os.path.join('model_data', 'testing.mat'))


def _main(args):
    os.chdir(os.path.join(os.getcwd(), 'yad2k-em3d'))
    # Parse input arguments
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    data_path = os.path.expanduser(args.data_path)
    test_results = os.path.expanduser(args.test_results)
    # Extract anchors, classes, images, and boxes from input files
    anchors = utils.get_anchors(anchors_path)
    classes = utils.get_classes(classes_path)
    images, boxes = utils.get_data(data_path)
    test_results = scipy.io.loadmat(test_results)
    t = test_results['output']
    cv2.imshow("TESTING", images[20])
    cv2.waitKey(0)
    ipdb.set_trace()
    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    drawn = utils.draw_boxes(images[0], t[:1], classes, class_names, scores=t[2])
    cv2.imshow('drawm', drawn)
    cv2.waitKey(0)
    # if you need to ensure the data being fed to the algorithm is correct, uncomment



if __name__ == '__main__':
    _main(parser.parse_args())
