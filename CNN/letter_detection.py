import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import ipdb
from helper import *

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

ipdb.set_trace()

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)