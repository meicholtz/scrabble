import cv2
import utils


for x in range(0, 100):
    img = utils.get_board('/Users/Alex/Desktop/Summer-2019/scrabble/labels.txt', x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    cv2.imshow("title", th3)
    cv2.waitKey(0)