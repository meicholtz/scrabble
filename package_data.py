from utils import *
import os
import argparse
import cv2
from colorama import Fore, Style

parser = argparse.ArgumentParser(description='Package data into a .npz file to use with YAD2K.')
parser.add_argument('-f', '--file', help='the file containing the labeled corners of the data',
                    type=str, default=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'))
parser.add_argument('-ld', '--label_directory', help='The path to a directory containing labeled image text files',
                    default=os.path.join(os.path.join(home(), 'labels')))
parser.add_argument('-n', '--name', help='name and path of the .npz file',
                    default='scrabble_dataset')
parser.add_argument('-s', '--size', type=int, help='the size to package the images. Must be divisible by 15.',
                    default=480)
# TODO: Make an 'all' option to package every file.
parser.add_argument('-num', '--num_files', help='Number of files you which to package', default=500)


def main(args):
    width, height = args.size, args.size
    assert width % 15 == 0, 'Width and height must be divisible by 15.'
    images = []
    labels = []
    num_files = args.num_files
    ld = args.label_directory
    file = args.file
    name = args.name
    imgs, pts = readlabels(file, ind='all')
    i = 0
    j = 0
    while(j < num_files):
        if(j % 10 == 0):
            print("{} of {}".format(j, num_files))
        textfile = os.path.basename(imgs[i])
        textfile = textfile.split('.')[0]
        textfile = textfile + '.txt'
        if(os.path.exists(os.path.join(ld, textfile))):
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            # warp the image
            img = imwarp(img, pts[i], sz=(width, height))
            images.append(img)
            # now open the label file and add the labels
            f = open(os.path.join(ld, textfile))
            temp = []
            # for each line in the file containing the coordinates of the boxes
            for line in f.readlines():
                # if the line is '~' which means a NONE label, skip it
                if (line.split(' ')[0] == '~'):
                    continue
                label = line.split(' ')
                # subtracting 65 from the value of the character allows for the classes to be one hot encoded
                # e.g. A = 0, B = 1, etc.
                label[0] = ord(label[0]) - 65
                # make everything in the label a float
                label = [float(i) for i in label]
                # make the class index (0) an int
                label[0] = int(label[0])
                temp.append(label)
            temp = np.asarray(temp)
            # if the length of the shape is 2, it indicates that at least one box was found and added to temp
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
    np.savez(name, images=images, boxes=labels)

if __name__ == '__main__':
    main(parser.parse_args())