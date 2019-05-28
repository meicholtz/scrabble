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

