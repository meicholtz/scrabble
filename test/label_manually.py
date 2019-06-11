import cv2
import utils
import os
import numpy as np
import re


MANUAL = True

BLANK_LABEL = 'NONE'


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


'''Main Function'''

# get the path of the labels text file
path = os.path.join(os.path.join(utils.home(), 'labels'), 'labels.txt')
instructions = "Press the letter on the keyboard that represents the letter shown. " \
               "\nIf no letter is show, press the spacebar to see the next tile."
print(instructions)
for ind in range(0, 100):
    imgname = utils.readlabels(path, ind)[0]
    imgname = os.path.basename(imgname)
    imgname = os.path.splitext(imgname)[0]
    txtfile = imgname + '.txt'
    txtfile = os.path.join(utils.home(), 'labels', txtfile)
    f = open(txtfile, "a+")
    file_length = file_len(txtfile)
    if (file_length > 224):
        continue
    start_row = np.floor(file_len / 15)
    if(start_row == 0):
        start_col = file_length
    else:
        start_col = file_length - 15
    # get the image of the board
    img = utils.get_board(path, ind)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = img.shape[0], img.shape[1]
    # get the individual tiles from the image
    sqs = utils.squares_from_img(img)
    # reshape to 4 dimensions so that x and y position can be tracked
    sqs = sqs.reshape((15, 15, sqs.shape[1], sqs.shape[2]))
    swh = sqs.shape[3]
    sq_width_height = float(swh / w)
    # for each tile:
    counter = 1
    for y in range(0,15):
        for x in range(0,15):
            center_x = x * swh
            center_x = float(center_x / w)
            center_y = y * swh
            center_y = float(center_y / h)
            text = manual_label(sqs[y][x])
            label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width_height, sq_width_height)
            f.write(label)
            print("Tile Number: {}".format(counter))
            counter += 1
    f.close()



