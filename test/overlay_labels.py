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

def main(args):
    # TODO: go through a directory of text files display instead of one file / image at a time
    ld = os.path.expanduser(args.labeldirectory)
    d = os.path.expanduser(args.datadirectory)
    assert os.path.isdir(ld), "{} is not a valid directory".format(ld)
    assert os.path.isdir(d), "{} is not a valid directory".format(d)
    directory = False
    if(args.name is None):
        directory = True
    name = args.name
    img_name = name + ".jpg"
    label_name = name + ".txt"
    f = open(os.path.join(ld, label_name))
    f.seek(0)
    img = cv2.imread(os.path.join(d, img_name))
    pts = np.array([[0.2547, 0.0854], [0.7984, 0.0583], [0.1500, 0.8979], [0.8781, 0.9042]])
    img = utils.imwarp(img, pts)
    overlay_text(img, f)

def overlay_text(img, lf):
    for line in lf.readlines():
        if(line.split(' ')[0] == 'NONE'):
            continue
        points = line.split(' ')
        x, y = float(points[1]), float(points[2])
        x, y = np.float32(x * img.shape[0]), np.float32(y * img.shape[0] + 55)
        cv2.putText(img=img, org=(x,y), text=points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255))
    cv2.imshow("yo", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main(parser.parse_args())