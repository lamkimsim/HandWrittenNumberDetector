import tkinter as tk


window = tk.Tk()
greeting = tk.Label(text="Python rocks!",
	 				width=20,
	 				height=20)

button = tk.Button(text="Predict",
				   width=5,
				   height=5)


greeting.pack()
button.pack()

window.mainloop()