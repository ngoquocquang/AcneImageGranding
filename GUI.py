from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from test import main

filenames = []
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Joint Acne Image Grading")
        self.minsize(250, 150)

        self.labelFrame = ttk.LabelFrame(self, text = "Select the Directory of Joint Ance Image that needed to grade")
        self.labelFrame.grid(column = 0, row = 0, padx = 125, pady = 75)

        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse Directory", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def fileDialog(self):
        self.filename = filedialog.askdirectory(initialdir = None, title = "Select A Directory")
        filenames= self.filename
        main(filenames)

root = Root()
root.mainloop()