import cv2
import utils
import ipdb
import os
import numpy as np
import PIL
import re
import io
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
    preprocess.org = preprocess.img
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 1000)
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

def overlay_text(img, lf):
    for line in lf.readlines():
        ipdb.set_trace()
        if(line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[1])
        cv2.putText(img=img, org=(x,y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.25,
        color=(0, 0, 255), thickness=1)
    cv2.namedWindow("text overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("text overlay", 1000, 1000)
    cv2.imshow("text overlay", img)
    cv2.waitKey(0)


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
    w, h = img.shape[0], img.shape[1]
    # convert img to be ocr'd
    img_2 = PIL.Image.fromarray(img)
    # config string for tesseract
    tessdata_dir_config = '--psm 11 -l eng'
    # get ocr label
    label = pytesseract.image_to_data(img_2, config=tessdata_dir_config)
    data = np.genfromtxt(io.BytesIO(label.encode()), delimiter="\t", skip_header=1, filling_values=1,
                         usecols=(6, 7, 8, 9, 10, 11), dtype=np.str)
    # delete all the rows where tesseract did not find a letter
    data = data[np.where(data[:, 4] != '-1')]
    # for each row, filter the label that tesseract returned
    for i in range(0, len(data)):
        text = filter_ocr(data[i][5])
        if (text == BLANK_LABEL):
            continue
        left = int(data[i][0])
        top = int(data[i][1])
        l_w = int(data[i][2])
        l_h = int(data[i][3])
        center_x, center_y = float(l_w / 2) + left, float(l_h / 2) + top
        center_x, center_y = float(center_x / w), float(center_y / h)
        sq_width, sq_height = float(l_w / w), float(l_h / h)
        label = "{} {} {} {} {} \n".format(text, center_x, center_y, sq_width, sq_height)
        f.write(label)

    overlay_text(img, f)
    f.close()




