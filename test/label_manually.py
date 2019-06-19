import cv2
import utils
import os
import numpy as np
import ipdb
import re
import argparse

BLANK_LABEL = 'NONE'

parser = argparse.ArgumentParser(description='Label Scrabble tiles manually')
parser.add_argument('-l', '--labeltextfile', type=str, help='the full path to the label text file',
                    default=os.path.join(os.path.join(utils.home(), 'labels'), 'labels.txt'))
parser.add_argument('-u', '--user', type=int, help='Which user? 0: Alexander 1: Dr. Eicholtz 2: Samantha',
                    default=0)


'''Main Function'''
def main(args):
    # get the path of the labels text file
    users = ["Alexander", "Dr. Eicholtz", "Samantha"]
    path = args.labeltextfile
    user = args.user
    if (user < 0 or user > 2):
        raise Exception("ERROR: User needs to be either 0, 1, or 2. 0: Alexander 1: Dr. Eicholtz 2: Samantha")

    instructions = "Current User: {} \n" \
                   "Press the letter on the keyboard that represents the letter shown. " \
                   "\nIf no letter is show, press the spacebar to see the next tile. " \
                   "\nPress ESC to exit.".format(users[user])
    print(instructions)
    counter = 1
    for ind in range(0, 10000):
        if(ind % 3 != user):
            continue
        imgname = utils.readlabels(path, ind)[0]
        imgname = os.path.basename(imgname)
        imgname = os.path.splitext(imgname)[0]
        txtfile = imgname + '.txt'
        txtfile = os.path.join(utils.home(), 'labels', txtfile)
        f = open(txtfile, "a+")
        if (not os.stat(txtfile).st_size == 0):
            continue
        # get the image of the board
        img = utils.get_board(path, ind)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("board", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("board", 400, 400)
        cv2.imshow("board", img)
        w, h = img.shape[0], img.shape[1]
        # get the individual tiles from the image
        sqs = utils.squares_from_img(img)
        # reshape to 4 dimensions so that x and y position can be tracked
        sqs = sqs.reshape((15, 15, sqs.shape[1], sqs.shape[2]))
        swh = sqs.shape[3]
        sq_width_height = float(swh / w)
        # for each tile:
        for y in range(0,15):
            for x in range(0,15):
                center_x = x * swh
                center_x = float(center_x / w)
                center_y = y * swh
                center_y = float(center_y / h)
                title = "Labeling tile: {}, boards completed: {}".format(counter % 225, np.floor(counter / 255))
                cv2.imshow(title, sqs[y][x])
                c = cv2.waitKey(0)
                # if the user hits escape quit the code
                if c == 27:
                    exit()
                text = filter_ocr(chr(c & 255))
                label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width_height, sq_width_height)
                f.write(label)
                counter += 1
                cv2.destroyWindow(title)
        f.close()
        cv2.destroyAllWindows()

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def manual_label(square):
    cv2.imshow("Press key on keyboard once to label", square)
    c = cv2.waitKey(0)
    # if the user hits escape quit the code
    if c == 27:
        exit()
    return filter_ocr(chr(c & 255))


def filter_ocr(text):
    if(text == '' or text == ' '):
        return BLANK_LABEL
    # grab the first character
    t = text[0]
    # remove non letter text
    regex = re.compile('[^a-zA-Z]')
    t = regex.sub('', t)
    if (t == ''):
        return BLANK_LABEL
    # capitalize letter
    t = t.capitalize()
    return t


def black_pixel_percentage(img):
    w = img.shape[1]
    return float(1 - cv2.countNonZero(img) / w**2)


if __name__ == '__main__':
    main(parser.parse_args())


