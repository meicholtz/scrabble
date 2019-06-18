import pytesseract
import PIL
import cv2
import numpy as np
import io
import re

BLANK_LABEL = 'NONE'


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


file_path = "/Users/Alex/Downloads/yes.jpg"
txtfile = "/Users/Alex/Downloads/testing.txt"
f = open(txtfile, "a+")
img = cv2.imread(file_path)
w, h = img.shape[0], img.shape[1]
# convert img to be ocr'd
img = PIL.Image.fromarray(img)
# config string for tesseract
tessdata_dir_config = '--psm 11 -l eng'
# get ocr label
label = pytesseract.image_to_data(img, config=tessdata_dir_config)
data = np.genfromtxt(io.BytesIO(label.encode()), delimiter="\t", skip_header=1, filling_values=1,
                     usecols=(6, 7, 8, 9, 10, 11), dtype=np.str)
# delete all the rows where tesseract did not find a letter
data = data[np.where(data[:, 4] != '-1')]
# for each row, filter the label that tesseract returned
for i in range(0, len(data)):
    text = filter_ocr(data[i][5])
    if(text == BLANK_LABEL):
        continue
    left = int(data[i][0])
    top = int(data[i][1])
    l_w = int(data[i][2])
    l_h = int(data[i][3])
    center_x, center_y = float(l_w/2) + left, float(l_h/2) + top
    center_x, center_y = float(center_x/w), float(center_y/h)
    sq_width, sq_height = float(l_w/w), float(l_h/h)
    label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width, sq_height)
    f.write(label)
f.close()


