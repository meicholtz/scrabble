import keras
from helper import *
import sys
import os
sys.path.insert(1, 'scrabble')
from utils import *

def sliding_window(img, model):
    assert img.shape[0] % 15 == 0, "Image shape must be divisible by 15"
    assert img.shape[1] % 15 == 0, "Image shape must be divisible by 15"
    # gather the squares from the image
    squares = squares_from_img(img)
    squares = squares / 255.0
    # reshape and resize the squares to work with Conv2D
    squares = squares.reshape(squares.shape + (-1,))
    squares = np.resize(squares, (squares.shape[0], 36, 36, 1))
    # have the model predict an output for each of the squares
    pred = model.predict(squares)
    print(pred)
    labels = []
    # for each prediction change the one hot encoding to a letter
    for p in pred:
        print(p)
        letter = np.argmax(p)
        # if the maximum is the last index, that means the model predicted a blank tile
        if letter == 26:
            # use ~ to represent a blank tile
            letter = '~'
            labels.append(letter)
            continue
        # 65 represents A
        letter = 65 + letter
        letter = chr(letter)
        labels.append(letter)
    return labels


filepath = "best_model.h5"
model = keras.models.load_model(filepath)

img = get_board(ind=24, file=os.path.join(home(), 'labels', 'labels.txt'))

labels = sliding_window(img, model)
ipdb.set_trace()

