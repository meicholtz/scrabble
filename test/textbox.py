import numpy as np
import cv2

font_scale = 0.8
font = cv2.FONT_HERSHEY_DUPLEX
margin = 20

# set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img = np.zeros((500, 500))
# set some text
text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = img.shape[1] // 2 - text_width // 2
text_offset_y = img.shape[0] // 2 - text_height // 2
# make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x - margin, text_offset_y + margin), (text_offset_x + text_width + margin, text_offset_y - text_height - margin))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
cv2.imshow("A box!", img)
cv2.waitKey(0)
