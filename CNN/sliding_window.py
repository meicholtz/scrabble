import keras
from helper import *
import sys
import os
sys.path.insert(1, 'scrabble')
from utils import *


def display_i(img, name="test"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sliding_window(img, model):
    assert img.shape[0] % 15 == 0, "Image shape must be divisible by 15"
    assert img.shape[1] % 15 == 0, "Image shape must be divisible by 15"
    # gather the squares from the image
    display_i(img)
    squares = squares_from_img(img)
    save_squares = squares
    squares = squares / 255.0
    squares = squares.reshape(squares.shape + (-1,))
    # have the model predict an output for each of the squares
    pred = model.predict(squares)
    print(pred)
    labels = []
    # for each prediction change the one hot encoding to a letter
    i = 0
    for p in pred:
        print(p)
        letter = np.argmax(p)
        # if the maximum is the last index, that means the model predicted a blank tile
        if letter == 26:
            # use ~ to represent a blank tile
            letter = '~'
            labels.append(letter)
            display_i(save_squares[i], str(letter))
            i += 1
            continue
        # 65 represents A
        letter = 65 + letter
        letter = chr(letter)
        display_i(save_squares[i], str(letter))
        i += 1
        labels.append(letter)
    return labels


filepath = "best_model2.h5"
model = keras.models.load_model(filepath)

img = get_board(ind=24, file=os.path.join(home(), 'labels', 'labels.txt'), sz=(540,540))
labels = sliding_window(img, model)
print(labels)


