#!/usr/bin/env python

'''Label sample Scrabble boards from warped images (batch version).

Use keyboard input to move around an image of a Scrabble board and label highlighted letters.
Valid key presses include arrow keys, Backspace (same as left arrow), space (same as right arrow),
a-z for labeling letters, 0 for clearing all current labels, Shift to skip the current board,
and Enter for saving the results to file.

You can also highlight a tile by clicking on it with the cursor.

    Required inputs
    ---------------
    labelfile   : string indicating the full path to a file containing a list
                  of image files and corresponding corners of a Scrabble board

    index       : integer indicating which line in the labelfile to process

    Optional inputs
    ---------------
    -s, -scroll : flag for automatically moving from left-to-right when a label
                  is applied
'''

import argparse
import os
from PIL import Image as PImage
from PIL import ImageTk
from graphics import *
from utils import *
from colorama import Fore, Style
import ipdb

parser = argparse.ArgumentParser(description='Label sample Scrabble boards from warped images (batch version).')
parser.add_argument('-l', '--labelfile', help='full path to label text file (see labeler.py)', default=os.path.join(home(), 'labels', 'labels.txt'))
parser.add_argument('-s', '--scroll', help='automatically move to next tile after labeling', action="store_true")
parser.add_argument('-u', '--user', help='name (full or partial) of user running this code', default='guest')

NUM_TILES = 15  # number of tiles on a Scrabble board
NUM_PIXELS = 36  # number of pixels per tile
BLANK_LABEL = '~'  # string for tiles that do not contain a letter


def main(args):
    # Parse input arguments
    labelfile = os.path.expanduser(args.labelfile)
    scrolling = args.scroll
    user, num_users = validateuser(args.user)

    # Start labeling boards
    flag = False  # flag to determine when user is ready to quit
    num_boards = linecount(labelfile)  # one line per board in label file
    for ind in range(num_boards):
        if flag:  # exit if user presses Escape
            break
        if ind % num_users != user:  # allow users to label simultaneously
            continue

        # Read data from labelfile
        imgfile, pts = readlabels(labelfile, ind)
        txtfile = jpg2txt(imgfile)  # for saving labels

        # Check to see if this image has already been labeled
        if os.path.exists(txtfile):
            print('{}WARNING: File already exists! Skipping to next image.\n{}{}'.format(Fore.YELLOW, txtfile, Style.RESET_ALL))
            continue

        # Process image
        img = cv2.imread(imgfile)
        img = imwarp(img, pts)
        sz = NUM_TILES * NUM_PIXELS
        img = cv2.resize(img, (sz, sz))

        # Initialize graphics window
        win = GraphWin(title=imgfile, width=sz, height=sz)
        win.setBackground(color_rgb(255, 255, 255))
        win.master.geometry("+50+50")  # move window to (50, 50) pixels on screen

        # Show current image
        I = ImageTk.PhotoImage(image=PImage.fromarray(img))
        win.create_image(0, 0, anchor='nw', image=I)
        win.update_idletasks()
        win.update()

        # Add text labels
        labels = []
        bg = []  # background for visibility
        for j in range(NUM_TILES):
            for i in range(NUM_TILES):
                anchor = Point((i + 0.2) * NUM_PIXELS, (j + 0.7) * NUM_PIXELS)
                txt = Text(anchor, '')
                txt.setSize(8)
                txt.setStyle('bold')
                txt.setTextColor(color_rgb(255, 0, 0))
                labels.append(txt)

                p1 = Point((i + 0.06) * NUM_PIXELS, (j + 0.56) * NUM_PIXELS)
                p2 = Point((i + 0.3) * NUM_PIXELS, (j + 0.8) * NUM_PIXELS)
                square = Rectangle(p1, p2)
                square.setFill(color_rgb(255, 225, 225))
                square.setOutline(color_rgb(255, 225, 225))
                bg.append(square)

        # Add rectangle to move around the board
        x, y = 0, 0
        margins = [0.5, 2.0]
        p1 = Point(x * NUM_PIXELS + margins[0], y * NUM_PIXELS + margins[0])
        p2 = Point((x + 1) * NUM_PIXELS - margins[1], (y + 1) * NUM_PIXELS - margins[1])
        rect = Rectangle(p1, p2)
        rect.setWidth(3)
        rect.setOutline(color_rgb(255, 218, 0))
        rect.draw(win)

        # Label until the user wants to quit
        while True:
            try:
                pt = win.checkMouse()
            except:
                break
            try:
                key = win.checkKey()
            except:
                break

            if pt:
                # Move the tile selector to where the user clicked
                col = int(pt.getX() // NUM_PIXELS)
                row = int(pt.getY() // NUM_PIXELS)
                rect.move(NUM_PIXELS * (col - x), NUM_PIXELS * (row - y))
                x, y = col, row

            if key in ['BackSpace', 'space', 'Left', 'Right', 'Up', 'Down']:
                # Move the tile selector around the board
                if key in ['BackSpace', 'Left']:
                    if x > 0:
                        x -= 1
                        rect.move(-NUM_PIXELS, 0)
                    else:
                        x = NUM_TILES - 1
                        if y > 0:
                            y -= 1
                            rect.move(NUM_PIXELS * (NUM_TILES - 1), -NUM_PIXELS)
                        else:
                            y = NUM_TILES - 1
                            rect.move(NUM_PIXELS * (NUM_TILES - 1), NUM_PIXELS * (NUM_TILES - 1))
                elif key in ['space', 'Right']:
                    if x < NUM_TILES - 1:
                        x += 1
                        rect.move(NUM_PIXELS, 0)
                    else:
                        x = 0
                        if y < NUM_TILES - 1:
                            y += 1
                            rect.move(-NUM_PIXELS * (NUM_TILES - 1), NUM_PIXELS)
                        else:
                            y = 0
                            rect.move(-NUM_PIXELS * (NUM_TILES - 1), -NUM_PIXELS * (NUM_TILES - 1))
                elif key == 'Up':
                    if y > 0:
                        y -= 1
                        rect.move(0, -NUM_PIXELS)
                    else:
                        y = NUM_TILES - 1
                        if x > 0:
                            x -= 1
                            rect.move(-NUM_PIXELS, NUM_PIXELS * (NUM_TILES - 1))
                        else:
                            x = NUM_TILES - 1
                            rect.move(NUM_PIXELS * (NUM_TILES - 1), NUM_PIXELS * (NUM_TILES - 1))
                elif key == 'Down':
                    if y < NUM_TILES - 1:
                        y += 1
                        rect.move(0, NUM_PIXELS)
                    else:
                        y = 0
                        if x < NUM_TILES - 1:
                            x += 1
                            rect.move(NUM_PIXELS, -NUM_PIXELS * (NUM_TILES - 1))
                        else:
                            x = 0
                            rect.move(-NUM_PIXELS * (NUM_TILES - 1), -NUM_PIXELS * (NUM_TILES - 1))

            elif key == 'Escape':
                win.close()
                flag = True
                break  # exit loop when 'Esc' is pressed

            elif key != '' and key in 'abcdefghijklmnopqrstuvwxyz':
                # Assign a letter to the current tile
                ind = x + y * NUM_TILES
                labels[ind].setText(key.upper())
                if not bg[ind].canvas:
                    bg[ind].draw(win)
                if not labels[ind].canvas:
                    labels[ind].draw(win)
                win.update()

                # Move to next tile if scrolling
                if scrolling:
                    if x == NUM_TILES - 1:
                        px = -NUM_PIXELS * (NUM_TILES - 1)
                        if y == NUM_TILES - 1:
                            x, y = 0, 0
                            rect.move(px, px)
                        else:
                            x = 0
                            y += 1
                            rect.move(px, NUM_PIXELS)
                    else:
                        x += 1
                        rect.move(NUM_PIXELS, 0)

            elif key == 'Delete':
                # Delete the label of the current tile
                ind = x + y * NUM_TILES
                labels[ind].setText('')
                labels[ind].undraw()
                bg[ind].undraw()
                win.update()

                # Move to next tile if scrolling
                if scrolling:
                    if x == NUM_TILES - 1:
                        px = -NUM_PIXELS * (NUM_TILES - 1)
                        if y == NUM_TILES - 1:
                            x, y = 0, 0
                            rect.move(px, px)
                        else:
                            x = 0
                            y += 1
                            rect.move(px, NUM_PIXELS)
                    else:
                        x += 1
                        rect.move(NUM_PIXELS, 0)

            elif key == '0':
                # Clear all current labels
                for i in range(len(labels)):
                    labels[i].setText('')
                    labels[i].undraw()
                    bg[i].undraw()

            elif key in ['Shift_L', 'Shift_R', 'Next']:
                # Skip to next file
                win.close()
                break

            elif key == 'Return':
                # Save labels to file
                save(txtfile, labels)
                win.close()
                break

            elif key != '':
                print(key)


def save(filename, labels):
    '''Save contents from image to file.'''
    f = open(filename, 'w')
    for i in range(len(labels)):
        row = i // NUM_TILES
        col = i - row * NUM_TILES
        letter = labels[i].getText()
        if letter == '':
            letter = '~'

        xmin = col / NUM_TILES
        ymin = row / NUM_TILES
        xmax = (col + 1) / NUM_TILES
        ymax = (row + 1) / NUM_TILES
        print("{:s} {:0.4f} {:0.4f} {:0.4f} {:0.4f}".format(letter, xmin, ymin, xmax, ymax), file=f)
        f.flush()
    print("Labels saved to file:", filename)
    f.close()


if __name__ == '__main__':
    main(parser.parse_args())
