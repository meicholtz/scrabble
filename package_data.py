import PIL
import numpy as np
from utils import *
import os
import argparse
import cv2
import ipdb

parser = argparse.ArgumentParser(description='Package data into a .npz file to use with YAD2K.')
parser.add_argument('-d', '--directory', type=str, help='the directory containing the image files', default=os.path.join(home(), 'data'))

parser.add_argument('-f', '--file', help='the file containing the labeled corners of the data', type=str, default=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'))
parser.add_argument('-n', '--name', help='name of the .npz file', default='my_data')
parser.add_argument('-o', '--output', help='where to save the .npz file', default=home())


IMAGE_SIZE = (825, 825)

def main(args):
    images = []
    labels = []
    dd = args.directory
    ld = os.path.join(home(), 'labels')
    imgs, pts = readlabels(os.path.join(ld, 'labels.txt'), ind='all')
    for i in range(len(imgs)):
        textfile = os.path.basename(imgs[i])
        textfile = textfile.split('.')[0]
        textfile = textfile + '.txt'
        if(os.path.exists(os.path.join(ld, textfile))):
            img = cv2.imread(imgs[i])
            # warp the image
            img = imwarp(img, pts[i], sz=IMAGE_SIZE)
            images.append(img)
            # now open the label file and add the labels
            f = open(os.path.join(ld, textfile))
            temp = []
            for line in f.readlines():
                if (line.split(' ')[0] == 'NONE'):
                    continue
                label = line.split(' ')
                label[0] = ord(label[0]) - 65
                label = label[:5]
                label = [float(i) for i in label]
                temp.append(label)
            labels.append(np.asarray(temp))
    labels = np.array(labels, dtype=object)
    images = np.array(images, dtype=np.uint8)
    np.savez("scrabble_dataset", images=images, boxes=labels)




if __name__ == '__main__':
    main(parser.parse_args())