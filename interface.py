from PIL import Image
from PIL import ImageDraw
from keras.models import load_model

import cv2
import tkinter as tk
import numpy as np
import pyscreenshot as ImageGrab
import os

model = load_model("LeNet.model")
filename = "Image.png"

# create a function that save the previous mouse coordinates
# to ahve a starting point to draw the line from
def draw_line(event):
	old_x, old_y = event.x, event.y

	canvas.create_line(old_x, old_y, event.x, event.y,
		width=20, fill="white",capstyle=tk.ROUND, smooth=True, splinesteps=36)

def clear_canvas(event):
	canvas.delete("all")
	textDisplay.delete(0, tk.END)
	probaDisplay.delete(0, tk.END)

def Predict(filename):
	textDisplay.delete(0, tk.END)
	probaDisplay.delete(0, tk.END)

	if filename:
		# save canvas in jpg
		x = window.winfo_rootx() + canvas.winfo_x()
		y = window.winfo_rooty() + canvas.winfo_y()
		x1 = x + canvas.winfo_width()
		y1 = y + canvas.winfo_height()
		ImageGrab.grab(bbox=(x,y,x1,y1)).save(filename)

		# convert to greyscale
		image = cv2.imread(filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image,(28,28))
		image_arr = np.array(image)
		image_arr = image_arr.reshape(1, 28, 28, 1)

		# predict class
		pred = model.predict_classes(image_arr)
		pred_proba = model.predict_proba(image_arr)
		pred_proba_max = np.amax(pred_proba)*100
		pred_proba_max = np.round(pred_proba_max, 1)

		# convert class to scalar
		predClass = pred[0]

		# display prediction
		textDisplay.insert(0, str(predClass))
		probaDisplay.insert(0, str(pred_proba_max)+"%")


# create window
window = tk.Tk()
window.geometry("1000x400")
window.title("Hand Written Digit Detector")

text_input = tk.StringVar()

textDisplay = tk.Entry(window, justify="right")
probaDisplay = tk.Entry(window, justify="right")


# instruction label
label = tk.Label(text="Draw on left canvas. Right click to clear.")
label.config(font=('helvetica', 14))

# prediction label
predictlabel = tk.Label(text="Predicted Number: ")
predictlabel.config(font=('helvetica', 14))

# proba label
probalabel = tk.Label(text="Probability: ")
probalabel.config(font=('helvetica', 14))

# create canvas
canvas = tk.Canvas(window, width=400, height=400,bg="black")

# left-click & drag to draw 
canvas.bind('<B1-Motion>', draw_line)

# right-click to clear canvas
canvas.bind('<Button-3>', clear_canvas)


# create buttons
predictBtn = tk.Button(window, text="Predict", command=lambda:Predict(filename))

#organise the elements
canvas.grid(row=0, column=0, rowspan=3)
label.grid(row=0, column=1, sticky=tk.N)
textDisplay.grid(row=1, column=1)
probaDisplay.grid(row=1, column=2)
predictlabel.grid(row=1, column=1, sticky=tk.W)
probalabel.grid(row=1, column=2, sticky=tk.W)
predictBtn.grid(row=2, column=1)


window.mainloop()