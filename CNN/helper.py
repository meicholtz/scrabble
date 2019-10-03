import numpy as np
import cv2
import sys
# access the base directory to be able to use scrabble/utils.py
sys.path.insert(1, 'scrabble')
from utils import *



def proccess_data(corner_labels=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'), class_labels=os.path.join(os.path.join(home(), 'labels'))):
    '''
    :param corner_labels: path to text file containing corner labels to perspective warp images
    :type corner_labels: string
    :param class_labels: path to text file containing coordinates of letter locations and letter labels
    :type class_labels: string
    :return:
    :rtype:
    '''
    width, height = 540, 540
    assert width % 15 == 0, 'Width and height must be divisible by 15.'
    images = []
    labels = []
    num_files = 10
    ld = class_labels
    file = corner_labels
    imgs, pts = readlabels(file, ind='all')
    i = 0
    j = 0
    while(j < num_files):
        if(j % 10 == 0):
            print("{} of {}".format(j, num_files))
        textfile = os.path.basename(imgs[i])
        textfile = textfile.split('.')[0]
        textfile = textfile + '.txt'
        if(textfile == 'EmptyBoard.txt'):
            continue
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
                labels.append([label[0]])
                xmin, ymin, xmax, ymax = label[1], label[2], label[3], label[4]
                # unnormalize
                xmin, ymin, xmax, ymax = xmin * width, ymin * width, xmax * width, ymax * width
                sq_width = xmax - xmin
                sq_height = ymax - ymin
                ipdb.set_trace()

            temp = np.asarray(temp)
            # if the length of the shape is 2, it indicates that at least one box was found and added to temp
            if(len(temp.shape) == 2):
                labels.append(temp)
                j += 1
            else:
                raise Exception("FOUND EMPTY FILE: {}".format(textfile))

        else:
            print("{}File does not have label: {} {}".format(Fore.YELLOW, textfile, Style.RESET_ALL))
        i += 1
    labels = np.array(labels, dtype=object)
    images = np.array(images, dtype=np.uint8)
    ipdb.set_trace()

proccess_data()