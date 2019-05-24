# USAGE
# python ocr.py --image images/example_01.png
# python ocr.py --image images/example_02.png  --preprocess blur

# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import utils
import os
import pdb

# todo: make path universal
path = '/Users/Alex/Desktop/Summer-2019/scrabble/labels.txt'
num_boards = 2
squares = utils.get_squares(path, num_boards)
squares = squares.reshape((-1,55,55))
for square in squares:
    cv2.imshow("s", square)
    cv2.waitKey(0)
    text = pytesseract.image_to_string(Image.fromarray(square))
    print(text)

# # load the example image and convert it to grayscale
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("Image", gray)
#
# # apply thresholding
# gray = cv2.threshold(gray, 0, 255,
# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# # blurring should be done to remove noise
# gray = cv2.medianBlur(gray, 3)
# img = []
#
# # load numpy images to be OCR'd
# text = pytesseract.image_to_string(Image.fromarray(img))
# print(text)
#
#
# # show the output images
# # cv2.imshow("Image", image)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)