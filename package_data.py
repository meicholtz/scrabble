import PIL
import numpy as np
from utils import *
import os
import argparse
import cv2
import ipdb
from colorama import Fore, Style

parser = argparse.ArgumentParser(description='Package data into a .npz file to use with YAD2K.')
parser.add_argument('-d', '--directory', type=str, help='the directory containing the image files', default=os.path.join(home(), 'data'))

parser.add_argument('-f', '--file', help='the file containing the labeled corners of the data', type=str, default=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'))
parser.add_argument('-n', '--name', help='name of the .npz file', default='my_data')
parser.add_argument('-o', '--output', help='where to save the .npz file', default=home())


IMAGE_SIZE = (825, 825)

def main(args):
    images = []
    labels = []
    letters, num_files = count_letters(count_boards=True)
    dd = args.directory
    ld = os.path.join(home(), 'labels')
    imgs, pts = readlabels(os.path.join(ld, 'labels.txt'), ind='all')
    i = 0
    j = 0
    while(j < num_files - 200):
        if(j % 10 == 0):
            print("{} of {}".format(j, num_files))
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
                if (line.split(' ')[0] == '~'):
                    continue
                label = line.split(' ')
                label[0] = ord(label[0]) - 65
                label = label[:5]
                label = [float(i) for i in label]
                temp.append(label)
            temp = np.asarray(temp)
            if(len(temp.shape) == 2):
                labels.append(temp)
                j += 1
            else:
                print("{}FOUND EMPTY FILE: {} {}".format(Fore.RED, textfile, Style.RESET_ALL))
        else:
            print("{}File does not have label: {} {}".format(Fore.YELLOW, textfile, Style.RESET_ALL))
        i += 1
    labels = np.array(labels, dtype=object)
    images = np.array(images, dtype=np.uint8)
    np.savez("YAD2K-master/model_data/scrabble_dataset", images=images, boxes=labels)
    data = np.load("YAD2K-master/model_data/scrabble_dataset.npz", allow_pickle=True)
    boxes = data['boxes']
    ipdb.set_trace()
    i =0
    for box in boxes:
        print(i, box.shape)
        if(len(box.shape) != 2):
            print("ERROR".format(Fore.RED, Style.RESET_ALL))
    ipdb.set_trace()

if __name__ == '__main__':
    main(parser.parse_args())