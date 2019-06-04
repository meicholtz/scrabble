import cv2
import utils
import pdb
import os
import numpy as np
import PIL
import pytesseract

# this is for sliders
def nothing(x):
    pass

# simple method to show an image
def show(img, title="show"):
    cv2.imshow(title, img)
    cv2.waitKey(0)

# take in an image, display it to the user with sliders for filtering the image
def preprocess(image):
    # get the width and height
    w, h = image.shape[0], image.shape[1]
    # make image BW
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the img
    img = cv2.resize(img, (w -150, h -150))
    cv2.namedWindow('image')
    # create trackbar for threshold and thinning parameters
    cv2.createTrackbar('Threshold', 'image', 60, 255, nothing)
    cv2.createTrackbar('Thinning', 'image', 0, 5, nothing)
    # create a slider that will invert the colors of the image
    cv2.createTrackbar('invert', 'image', 0, 1, nothing)
    flag = True
    kernel = np.ones((2, 2), np.uint8)
    while(1):
        # check the positions of the trackbars and store them
        x = cv2.getTrackbarPos('Threshold', 'image')
        y = cv2.getTrackbarPos('Thinning', 'image')
        invert = cv2.getTrackbarPos('invert', 'image')
        # apply filtering with trackbar parameters
        if(invert == 1 and flag):
            img = 255 - img
            flag = False
        if(invert == 0):
            flag = True
        ret, temp = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
        temp2 = cv2.erode(temp, kernel, iterations=y)
        # display filtered image
        cv2.imshow('image', temp2)
        # keep the window alive unless escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # if the user presses D stop the preprocessing and return the processed image
        if k == ord('d'):
            return temp2

def ocr(square):
    # convert square to be ocr'd
    img = PIL.Image.fromarray(square)
    # get ocr label
    label = pytesseract.image_to_string(img, config='--psm 10')
    print(label)
    show(square)


'''Main Function'''
# get the path of the labels text file
path = os.path.join(os.path.dirname(os.getcwd()), 'labels.txt')
# get the image of the board
test2 = utils.get_board(path, 2)
# preprocess the image
img = preprocess(test2)
# display the processed image
show(img, title=str(2))
# get the individual tiles from the image
sqs = utils.squares_from_img(img)
# for each tile:
for s in sqs:
    # get width and height
    w,h = s.shape[0], s.shape[1]
    # enlarge the tile by a factor of 10
    s = cv2.resize(s, (w*10, h*10))
    # get the ocr of the tile
    ocr(s)


# for x in range(0, 100):
#     test2 = utils.get_board(path, x)
#     img = preprocess(test2)
#     show(img, title=str(x))
#     cv2.destroyAllWindows()
#     if(x == 2):
#         sqs = utils.squares_from_img(img)
#         for s in sqs:
#             ocr(s)




