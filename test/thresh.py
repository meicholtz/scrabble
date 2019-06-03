import cv2
import utils
import pdb
import os
import numpy as np


def nothing(x):
    pass

def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)

def preprocess(image):
    # get the width and height
    w, h = image.shape[0], image.shape[1]
    # make image BW
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the img
    img = cv2.resize(img, (w -150, h -150))
    cv2.namedWindow('image')
    # create two trackbars for threshold parameters
    cv2.createTrackbar('Black Threshold', 'image', 60, 255, nothing)
    cv2.createTrackbar('White Threshold', 'image', 255, 255, nothing)
    # create a slider that will invert the colors of the image
    cv2.createTrackbar('invert', 'image', 0, 1, nothing)
    while(1):
        # check the positions of the trackbars and store them
        x = cv2.getTrackbarPos('Black Threshold', 'image')
        y = cv2.getTrackbarPos('White Threshold', 'image')
        invert = cv2.getTrackbarPos('invert', 'image')
        if(invert == 1):
            img = (255 - img)
        # apply filtering with trackbar parameters
        ret, temp = cv2.threshold(img, x, y, cv2.THRESH_BINARY)
        # display filtered image
        cv2.imshow('image', temp)
        # keep the window alive unless escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


for x in range(0, 100):
    path = os.path.join(os.path.dirname(os.getcwd()), 'labels.txt')
    test = utils.get_board(path, x, squares=True)
    test2 = utils.get_board(path, x)
    # utils.showboard(cv2.Canny(test2, 20, 50))
    # img = cv2.Canny(test2, 20, 50)
    preprocess(test2)
    #
    # for y in range(len(test)):
    #     preprocess(test[y])
    # img = utils.get_board(path, x)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    # sqs = utils.squares_from_img(th3)



