import numpy as np
import os
import ipdb
from utils import *


''' This file is to fix previous mistakes when labeling files. '''


# open the directory with text files
directory = os.path.join(home(), 'labels')
for file in os.listdir(directory):
    if(file != 'labels.txt' and file != 'labels1.txt' and file.endswith('.txt')):
        f = open(os.path.join(directory, file), "r")
        lines = f.readlines()
        for i in range(0, len(lines)):
            a = lines[i].split(' ')
            if(a[0] == 'NONE'):
                a[0] = '~'
            x, y = float(a[1]), float(a[2])
            swh = float(a[3])
            x_max, y_max = x + swh, y + swh
            lines[i] = '{0} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(a[0], x, y, x_max, y_max)
        with open(os.path.join(directory, file), 'w') as f:
                f.writelines(lines)
        f.close()
