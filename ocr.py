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
import matplotlib.pyplot as plt


path = os.path.join(os.getcwd(),'labels.txt')
num_boards = 2
board = utils.get_board(path, 1)
plt.imshow(board)
plt.show()

