from utils import *
import os
import numpy as np


letters = np.zeros(26)
label_files = ['labels.txt', 'labels1.txt']
directory = os.path.join(home(), 'labels')
for filename in os.listdir(directory):
    if filename.endswith(".txt") and filename not in label_files:
        f = open(os.path.join(directory, filename))
        for line in f.readlines():
            letter = line.split(' ')[0]
            print(letter)
            if(letter == 'NONE' or ord(letter) < 65):
                continue
            letters[ord(letter) - 65] += 1
    else:
        continue
print(letters)
