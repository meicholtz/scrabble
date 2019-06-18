import cv2
import utils
import ipdb
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Input a text file and an image to overlay labels on the image')
parser.add_argument('-ld', '--labeldirectory', type=str, help='the directory containing the label text file',
                    default=os.path.join(utils.home(), 'labels'))
parser.add_argument('-d', '--datadirectory', type=str, help='the directory containing the image',
                    default=os.path.join(utils.home(), 'data'))
parser.add_argument('-n', '--name', type=str, help='the name of an image with no extension ex: Photo_2005-08-20_006 '
                                                   'leave empty to cycle through a directory')



def overlay_text(img, lf):
    for line in lf.readlines():
        if(line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[1])
        cv2.putText(img=img, org=(x,y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=10,
        color=(0, 0, 255), thickness=7)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1000, 1000)
    cv2.imshow("img", img)
    cv2.waitKey(0)

f = open("/Users/Alex/Downloads/testing.txt")
f.seek(0)
img = cv2.imread("/Users/Alex/Downloads/yes.jpg")
overlay_text(img, f)
