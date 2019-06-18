import cv2
import utils
import ipdb
import os
import numpy as np
import PIL
import re
import pytesseract


MANUAL = True

BLANK_LABEL = 'NONE'

def ocr(img):
    print("Individual square: width: {}, height: {}".format(img.shape[0], img.shape[1]))
    # convert square to be ocr'd
    img = PIL.Image.fromarray(img)
    # config string for tesseract
    tessdata_dir_config = '--tessdata-dir "/usr/local/Cellar/tesseract/4.0.0_1/share/tessdata" --psm 10  --oem 2 '
    # get ocr label
    label = pytesseract.image_to_string(img, config=tessdata_dir_config)
    return filter_ocr(label)


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
txtfile = os.path.join(utils.home(), 'labels', 'testing.txt')
f = open(txtfile, "a+")
if (not os.stat(txtfile).st_size == 0):
    exit()
# get the image of the board
img = cv2.imread("/Users/Alex/Desktop/Summer-2019/scrabble/data/testing.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
w, h = img.shape[0], img.shape[1]
print("full image: width: {}, height: {}".format(w, h))
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
        text = ocr(sqs[y][x])
        label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width_height, sq_width_height)
        f.write(label)
        print("Tile Number: {}".format(counter))
        counter += 1
f.close()



