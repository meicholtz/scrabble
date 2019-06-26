import numpy as np
import os
import ipdb
from utils import *


''' This file is to fix previous mistakes when labeling files. '''


# open the directory with text files
directory = os.path.join(home(), 'labels')
for file in os.listdir(directory):
    if(file != 'labels.txt' and file != 'labels1.txt' and file.endswith('.txt')):
        f = open(file)
        lines = f.readlines()
        for i in range(0, len(lines)):
            a = lines[i].split(' ')
            if(a[0] == 'NONE'):
                a[0] = '~'
            x, y = float(a[1]), float(a[2])
