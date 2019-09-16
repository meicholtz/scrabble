#! /usr/bin/env python
""""""
import os

import ipdb
import numpy as np
import cv2


def check_data(images, boxes):
    assert images.shape[1] % 15 == 0 and images.shape[2] % 15 == 0, "Width / Height of images is not divisible by 15."
    square_size = images.shape[1] / 15
    for i in range(len(images)):  # for every image in the images
        img = images[i]
        for box in boxes[i]:
            x, y = box[0], box[1]
            wd, ht = box[2], box[3]
            ipdb.set_trace()
            # the points are normalized and to reverse that multiply by the image shape.
            x, y, wd, ht = np.float32(x * img.shape[0]), np.float32(y * img.shape[0]), \
                         np.float32(wd * img.shape[0]), np.float32(ht * img.shape[0])
            ipdb.set_trace()
            # the classes of the packaged data contained number values for each letter with 'A' being 0. To get a
            # letter, take the number and add 65
            letter = chr(int(box[4]) + 65)
            cv2.putText(img=img, org=(x, y), text=letter, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255))
        cv2.namedWindow("Text Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Text Overlay", 1000, 1000)
        cv2.imshow("Text Overlay", img)
        cv2.waitKey(0)
