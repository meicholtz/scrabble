import cv2
import utils
import pdb
import os
import numpy as np
import PIL
import pytesseract

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
    kernel = np.ones((3, 3), np.uint8)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if (preprocess.invert):
                image = 255 - image
            ret, temp = cv2.threshold(image, x, 255, cv2.THRESH_BINARY)
            temp2 = cv2.dilate(temp, kernel, iterations=y)
            return temp2



def ocr(square):
    # convert square to be ocr'd
    img = PIL.Image.fromarray(square)
    # config string for tesseract
    tessdata_dir_config = '--tessdata-dir "/usr/local/Cellar/tesseract/4.0.0_1/share/tessdata" --psm 10  --oem 2 '
    # get ocr label
    label = pytesseract.image_to_string(img, config=tessdata_dir_config)
    return label


'''Main Function'''

# get the path of the labels text file
path = os.path.join(os.path.dirname(os.getcwd()), 'labels.txt')
for ind in range(0, 100):
    # get the image of the board
    test2 = utils.get_board(path, ind)
    w, h = test2.shape[0], test2.shape[1]
    # preprocess the image
    img = preprocess(test2)
    cv2.destroyAllWindows()
    # get the individual tiles from the image
    sqs = utils.squares_from_img(img)
    # for each tile:
    sq = 1
    for s in sqs:
        # get width and height
        s_w,s_h = s.shape[0], s.shape[1]
        # get the float values for width and height
        x, y = float(s_w / w), float(s_h / h)
        
        # get the ocr of the tile
        print(ocr(s))
        cv2.imshow("square", s)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break


