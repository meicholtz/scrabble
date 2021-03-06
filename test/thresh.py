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
# callback function for Threshold and Thinning sliders
def nothing(x):
    pass

# callback function for Invert slider
def invert_img(x):
    preprocess.img = 255 - preprocess.img
    preprocess.invert = not preprocess.invert
    print(preprocess.invert)

# callback function for Adaptive Threshold slider
# x is the value of the slider
def adaptive_threshold(x):
    # TODO: KNOWN ISSUE - adaptive threshold undoes inversion of image and messes up the invert flag.
    # Inversion must be done after adaptive thresholding
    # preprocess.org is the original image before processing
    # preprocess.img is how the image gets displayed
    if x == 0:
        preprocess.img = preprocess.org
    elif x == 1:
        preprocess.img = cv2.adaptiveThreshold(preprocess.org, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)
    elif x == 2:
        preprocess.img = cv2.adaptiveThreshold(preprocess.org, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                               cv2.THRESH_BINARY, 11, 2)


# take in an image, display it to the user with sliders for filtering the image
def preprocess(image):
    # get the width and height
    w, h = image.shape[0], image.shape[1]
    # make image BW
    preprocess.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the img
    preprocess.img = cv2.resize(preprocess.img, (w - 250, h - 250))
    preprocess.org = preprocess.img
    preprocess.img = cv2.equalizeHist(preprocess.img)
    cv2.namedWindow('image')
    # create trackbar for threshold and thinning parameters
    cv2.createTrackbar('Adaptive Threshold', 'image', 0, 2, adaptive_threshold)
    cv2.createTrackbar('Threshold', 'image', 60, 255, nothing)
    cv2.createTrackbar('Thinning', 'image', 0, 5, nothing)
    # create a slider that will invert the colors of the image
    cv2.createTrackbar('Invert', 'image', 0, 1, invert_img)
    kernel = np.ones((2, 2), np.uint8)
    preprocess.invert = False
    while True:
        # check the positions of the trackbars and store them
        x = cv2.getTrackbarPos('Threshold', 'image')
        y = cv2.getTrackbarPos('Thinning', 'image')
        ret, temp = cv2.threshold(preprocess.img, x, 255, cv2.THRESH_BINARY)
        temp2 = cv2.dilate(temp, kernel, iterations=y)
        # display filtered image
        cv2.imshow('image', temp2)
        # keep the window alive unless escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # if the user presses D stop the preprocessing and return the processed image
        if k == ord('d'):
            if preprocess.invert:
                image = 255 - image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            t = cv2.getTrackbarPos('Adaptive Threshold', 'image')
            if t == 1:
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                      cv2.THRESH_BINARY, 11, 2)
                temp2 = cv2.dilate(image, kernel, iterations=y)
            elif t == 2:
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                       cv2.THRESH_BINARY, 11, 2)
                temp2 = cv2.dilate(image, kernel, iterations=y)
            else:
                ret, temp = cv2.threshold(image, x, 255, cv2.THRESH_BINARY)
                temp2 = cv2.dilate(temp, kernel, iterations=y)
            # if (preprocess.invert):
            #     temp2 = 255 - temp2
            return temp2

def manual_label(square):
    cv2.imshow("Press key on keyboard once to label", square)
    if (black_pixel_percentage(square) < 0.05):
        return BLANK_LABEL
    c = cv2.waitKey(0)
    return filter_ocr(chr(c & 255))

def ocr(square):
    if black_pixel_percentage(square) > .25:
        return BLANK_LABEL
    # convert square to be ocr'd
    img = PIL.Image.fromarray(square)
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
for ind in range(0, 100):
    imgname = utils.readlabels(path, ind)[0]
    imgname = os.path.basename(imgname)
    imgname = os.path.splitext(imgname)[0]
    txtfile = imgname + '.txt'
    txtfile = os.path.join(utils.home(), 'labels', txtfile)
    f = open(txtfile, "a+")
    if (not os.stat(txtfile).st_size == 0):
        continue
    # get the image of the board
    test2 = utils.get_board(path, ind)
    w, h = test2.shape[0], test2.shape[1]
    # pre-process the image
    img = preprocess(test2)
    utils.imshow(img)
    cv2.destroyAllWindows()
    # get the individual tiles from the image
    sqs = utils.squares_from_img(img)
    # reshape to 4 dimensions so that x and y position can be tracked
    sqs = sqs.reshape((15,15,sqs.shape[1],sqs.shape[2]))
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
            if (MANUAL):
                text = manual_label(sqs[y][x])
            else:
                text = ocr(sqs[y][x])
            label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width_height, sq_width_height)
            f.write(label)
            print("Tile Number: {}".format(counter))
            counter += 1
    f.close()



