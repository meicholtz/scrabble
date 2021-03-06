import cv2
import utils
import ipdb
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import io
import pytesseract

MANUAL = True

BLANK_LABEL = 'NONE'

def imshow_components(labels):
    ipdb.set_trace()
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    cv2.namedWindow("CCA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCA", 1000, 1000)
    cv2.imshow("CCA", labeled_img)
    cv2.waitKey()

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
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 500, 500)
    # get the width and height
    w, h = image.shape[0], image.shape[1]
    # make image BW
    preprocess.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocess.org = preprocess.img
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 1000)
    # create trackbar for threshold and thinning parameters
    cv2.createTrackbar('Adaptive Threshold', 'image', 0, 2, adaptive_threshold)
    cv2.createTrackbar('Thinning', 'image', 0, 5, nothing)
    cv2.createTrackbar('Minimum Pixels', 'image', 0, 500, nothing)
    cv2.createTrackbar('Maximum Pixels', 'image', 500, 500, nothing)
    # create a slider that will invert the colors of the image
    cv2.createTrackbar('Invert', 'image', 0, 1, invert_img)
    kernel = np.ones((2, 2), np.uint8)
    preprocess.invert = False
    while True:
        # check the positions of the trackbars and store them
        y = cv2.getTrackbarPos('Thinning', 'image')
        min = cv2.getTrackbarPos('Minimum Pixels', 'image')
        max = cv2.getTrackbarPos('Maximum Pixels', 'image')
        temp2 = cv2.erode(preprocess.img, kernel, iterations=y)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(temp2, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        # your answer image
        img2 = np.zeros((output.shape))
        # testing = np.uint8(output)
        # for every component in the image, you keep it only if it's above min_size
        # TODO: This for loop slows down the program significantly
        for i in range(0, nb_components):
            if sizes[i] >= min and sizes[i] <= max:
                # print(sizes[i])
                # testing[output != i + 1] = 255
                # utils.imshow(testing)
                img2[output == i + 1] = 255
            # testing = np.uint8(output)
        # display filtered image
        cv2.imshow('image', img2)
        cv2.imshow('original', preprocess.org)
        # keep the window alive unless escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # if the user presses D stop the preprocessing and return the processed image
        if k == ord('d'):
            cv2.destroyAllWindows()
            img2 = 255 - img2
            return img2

def overlay_text(img, labels):
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Roboto-Regular.ttf', size=45)
    color = 'rgb(206, 35, 35)'
    for line in labels:
        if (line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[1])
        cv2.putText(img=img, org=(x, y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 0), thickness=1)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for line in labels:
        if(line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[1])
        cv2.putText(img=img, org=(x,y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0), thickness=1)
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
    img_2 = Image.fromarray(img)
    # config string for tesseract
    tessdata_dir_config = '--psm 11 -l eng'
    # get ocr label
    label = pytesseract.image_to_data(img_2, config=tessdata_dir_config)
    data = np.genfromtxt(io.BytesIO(label.encode()), delimiter="\t", skip_header=1, filling_values=1,
                         usecols=(6, 7, 8, 9, 10, 11), dtype=np.str)
    # delete all the rows where tesseract did not find a letter
    data = data[np.where(data[:, 4] != '-1')]
    # for each row, filter the label that tesseract returned
    labels = []
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
        labels.append(label)
        f.write(label)
    overlay_text(img, labels)
    ipdb.set_trace()
    f.close()




