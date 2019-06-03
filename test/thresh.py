import cv2
import utils
import pdb
import os

for x in range(0, 100):
    path = os.path.join(os.path.dirname(os.getcwd()), 'labels.txt')
    img = utils.get_board(path, x)
    # t = utils.get_board(path, x, squares=True)
    # t = utils.squares_from_img(img)
    # cv2.imshow("square", t[0])
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imshow("title", th3)
    cv2.waitKey(0)
    sqs = utils.squares_from_img(th3)
    cv2.imshow("thresh square", sqs[0])
    cv2.waitKey(0)
