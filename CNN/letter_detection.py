import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
import numpy as np
import ipdb
from helper import *
import sys
import os 
sys.path.insert(1, 'scrabble')
from utils import *


num_classes = 27
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(11502, 36, 36), padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

data = np.load(os.path.join(home(), "CNN", "data.npz"), allow_pickle=True)

X, Y = data["X"], data["Y"]

# Conv2d requires 4 dim numpy array
X = X.reshape((-1,) + X.shape)
# Convert labels to categorical one-hot encoding
Y = keras.utils.to_categorical(Y, num_classes=num_classes)

X_train, Y_train = X[:11000], Y[:11000]
X_test, Y_test = X[11000:], Y[11000:]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=10)