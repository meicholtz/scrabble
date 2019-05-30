import tkinter as tk
from PIL import Image, ImageTk
import cv2
import utils

def change_pic():
    vlabel.configure(image=root.photo1)
    print("updated")

root = tk.Tk()

img = utils.get_board('/Users/Alex/Desktop/Summer-2019/scrabble/labels.txt', 0)
photo = Image.fromarray(img)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
root.photo = ImageTk.PhotoImage(photo)
root.photo1 = ImageTk.PhotoImage(Image.fromarray(grayImage))

vlabel=tk.Label(root,image=root.photo)
vlabel.pack()

b2=tk.Button(root,text="Threshold",command=change_pic)
b2.pack()

root.mainloop()