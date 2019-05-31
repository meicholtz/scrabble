import tkinter as tk
from PIL import Image, ImageTk
import cv2
import utils
import numpy as np
import pdb

def change_pic():
    global img
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = Image.fromarray(grayImage)
    vlabel.configure(image=ImageTk.PhotoImage(temp))
    print("updated")

root = tk.Tk()
img = utils.get_board('/Users/Alex/Desktop/Summer-2019/scrabble/labels.txt', 0)
photo = Image.fromarray(img)

root.photo = ImageTk.PhotoImage(photo)
vlabel=tk.Label(root,image=root.photo)
vlabel.pack()

b2=tk.Button(root,text="BW",command=change_pic)
b2.pack()

root.mainloop()