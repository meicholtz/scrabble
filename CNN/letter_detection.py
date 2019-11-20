import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
import numpy as np
import ipdb
from helper import *
import sys
import os 
sys.path.insert(1, 'scrabble')
from utils import *


def display_i(img, name="test"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

num_classes = 27

data = np.load(os.path.join(home(), "CNN", "data.npz"), allow_pickle=True)

X, Y = data["X"], data["Y"]

X = X / 255.0
# Conv2d requires 4 dim numpy array and since the backend of keras is using channels
# last, append the 4th dimension to the end
X = X.reshape(X.shape + (-1,))

# Convert labels to categorical one-hot encoding
Y = keras.utils.to_categorical(Y, num_classes=num_classes)

# use the first 11000 images for training and the last 500 images for testing
X_train, Y_train = X[:7000], Y[:7000]
X_test, Y_test = X[7000:], Y[7000:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=X_train.shape[1:], padding="valid"))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(num_classes, activation='softmax'))


# # load weights
# model.load_weights("weights.best.hdf5")

# opt = SGD(lr=0.001)

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# Train the model, iterating on the data in batches of 32 samples

# checkpoint
model_filepath="best_model.h5"
model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max', save_weights_only=False)
wts_filepath="weights.best.hdf5"
wts_checkpoint = ModelCheckpoint(wts_filepath, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [model_checkpoint, wts_checkpoint]

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=32, callbacks=callbacks_list)