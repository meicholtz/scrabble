import cv2
import utils
import pdb

for x in range(0, 100):
    img = utils.get_board('/Users/Alex/Desktop/Summer-2019/scrabble/labels.txt', x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imshow("title", th3)
    cv2.waitKey(0)
    sqs = utils.squares_from_img(img)
    utils.display_board(sqs)
