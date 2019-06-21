import PIL
import numpy as np
from utils import *
import os
import argparse
import ipdb

parser = argparse.ArgumentParser(description='Package data into a .npz file to use with YAD2K.')
parser.add_argument('-d', '--directory', type=str, help='the directory containing the image files', default=os.path.join(home(), 'data'))

parser.add_argument('-f', '--file', help='the file containing the labeled corners of the data', type=str, default=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'))
parser.add_argument('-n', '--name', help='name of the .npz file', default='my_data.npz')
parser.add_argument('-o', '--output', help='where to save the .npz file', default=home())


IMAGE_SIZE = (825, 825)

def main(args):
    images = []
    labels = []
    dd = args.directory
    ld = os.path.join(home(), 'labels')
    imgs, pts = readlabels(os.path.join(ld, 'labels.txt'), ind='all')
    ipdb.set_trace()




if __name__ == '__main__':
    main(parser.parse_args())