import numpy as np
import cv2
import sys
# access the base directory to be able to use scrabble/utils.py
sys.path.insert(1, 'scrabble')
from utils import *


def display_i(img, name="test"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def proccess_data(corner_labels=os.path.join(os.path.join(home(), 'labels'), 'labels.txt'),
                  class_labels=os.path.join(os.path.join(home(), 'labels')), num_files=600):
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
    ld = class_labels
    file = corner_labels
    imgs, pts = readlabels(file, ind='all')
    i = 0
    j = 0
    print("-----CONVERTING IMAGES AND LABELS-----")
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
            squares = squares_from_img(img)
            # for s in squares:
            #     images.append(s)
            # now open the label file and add the labels
            f = open(os.path.join(ld, textfile))
            # for each line in the file containing the coordinates of the boxes
            s = 0
            for line in f.readlines():
                # if the line is '~' which means a NONE label, assign it 26
                if (line.split(' ')[0] == '~'):
                    labels.append([26])
                    images.append(squares[s])
                    s += 1
                    continue
                label = line.split(' ')
                # subtracting 65 from the value of the character allows for the classes to be one hot encoded
                # e.g. A = 0, B = 1, etc.
                label[0] = ord(label[0]) - 65
                # make the class index (0) an int
                label[0] = int(label[0])
                labels.append([label[0]])
                images.append(squares[s])
                s += 1
        else:
            print("{}File does not have label: {} {}".format(Fore.YELLOW, textfile, Style.RESET_ALL))
        i += 1
        j += 1
    labels = np.array(labels, dtype=object)
    images = np.array(images, dtype=np.uint8)
    return images, labels


def balance_data(images, labels):
    X = []
    Y = []
    # bincount only works with flat integer arrays
    flat = labels.flatten()
    flat = flat.astype(np.int)
    # bincount returns the frequency of each label
    freq = np.bincount(flat)
    # find the lowest number of occurrences
    num = np.min(freq)
    # this number will be used to balance all the labels.
    # All labels will have the same number of entries for the CNN
    for i in range(27):
        ind = np.where(labels == [i])[0]
        chosen = np.random.choice(ind, num)
        # if this is the first iteration
        if(i==0):
            X = images[chosen]
            Y = labels[chosen]
        else:
            X = np.concatenate((X, images[chosen]))
            Y = np.concatenate((Y, labels[chosen]))
    return X, Y


def create_files():
    X, Y = proccess_data()
    X, Y = balance_data(X, Y)
    np.savez(os.path.join(home(), "CNN", "data.npz"), X=X, Y=Y)

