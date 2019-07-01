#!/usr/bin/env python

'''Get basic statistics about labeled training data.'''

from utils import *
import os
import numpy as np

BLANK_LABEL = '~'  # string for tiles that do not contain a letter


def main():
    root = os.path.join(home(), 'labels')  # directory containing labels
    labelfiles = ['labels.txt', 'labels1.txt']  # do not include in statistics

    # Compute count of each letter
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    cnt = {key: 0 for key in letters}
    for file in os.listdir(root):
        if file.endswith(".txt") and file not in labelfiles:
            f = open(os.path.join(root, file))
            for line in f.readlines():
                letter = line.split(' ')[0]
                if letter != BLANK_LABEL and letter in letters:
                    cnt[letter] += 1

    # Print output
    for key in cnt:
        print("{} {}".format(key, cnt[key]))


if __name__ == '__main__':
    main()
