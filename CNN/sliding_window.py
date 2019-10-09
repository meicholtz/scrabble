import keras
from helper import *
import sys
import os
sys.path.insert(1, 'scrabble')
from utils import *

def sliding_window(img, model):
    assert img.shape[0] % 15 == 0, "Image shape must be divisible by 15"
    assert img.shape[1] % 15 == 0, "Image shape must be divisible by 15"
    squares = squares_from_img(img)
    ipdb.set_trace()
    pred = model.predict(squares[0])
    ipdb.set_trace()

filepath = "best_model.h5"
model = keras.models.load_model(filepath)

img = get_board(ind=24, file=os.path.join(home(), 'labels', 'labels.txt'))

sliding_window(img, model)


