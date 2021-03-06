#!/usr/bin/env python

'''Get basic statistics about labeled training data.'''

from utils import *
import os
import numpy as np

BLANK_LABEL = '~'  # string for tiles that do not contain a letter


def main():
    root = os.path.join(home(), 'labels')  # directory containing labels
    skip = ['labels.txt', 'labels1.txt']  # do not include in statistics

    # Compute count of each letter
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    cnt = {key: 0 for key in letters}
    boards = 0
    for file in os.listdir(root):
        if file.endswith(".txt") and file not in skip:
            boards += 1
            with open(os.path.join(root, file)) as f:
                for line in f.readlines():
                    letter = line.split()[0]
                    if letter != BLANK_LABEL and letter in letters:
                        cnt[letter] += 1

    # Print output
    for key in cnt:
        print("{} {}".format(key, cnt[key]))
    print("=====================")
    total = sum(cnt.values())
    print(total, "total letters")
    print(total // boards, "letters per board")


if __name__ == '__main__':
    main()
