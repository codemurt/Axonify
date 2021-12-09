import inspect
import os
import sys
import tkinter as tk
import tkinter.ttk as ttk
from time import sleep
from tkinter.filedialog import askdirectory

import train


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="DCGAN", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        tk.Button(self, text="Generate", command=lambda: controller.show_frame("GeneratorPage")).pack()
        tk.Button(self, text="Train", command=lambda: controller.show_frame("DatasetSelection")).pack()


class GeneratorPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Select the number of images to generate:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        inp = tk.Entry(self)
        inp.pack()

        def generate_pics(imgCount):
            if imgCount.isdigit():
                print("WIP")
                #
                # do stuff
                #
                inp.delete(0, 'end')
                controller.show_frame("GenerationEnd")

        def show_start():
            inp.delete(0, 'end')
            controller.show_frame("StartPage")

        tk.Button(self, text="Generate", command=lambda: generate_pics(inp.get())).pack()
        tk.Button(self, text="Back", command=show_start).pack()


class GenerationEnd(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Ready! Your pictures are saved in the "Results" folder.').pack()
        tk.Button(self, text='Return to menu', command=lambda: controller.show_frame("StartPage")).pack()


class DatasetSelection(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Select dataset:').pack()

        def get_datasets():
            print("WIP")
            out = ()
            if not os.path.exists("Resources/datasets.txt"):
                with open("Resources/datasets.txt", "w") as f:
                    f.write("")
            datasets_file = open("Resources/datasets.txt", 'r')
            global datasets
            datasets = datasets_file.readlines()
            datasets_file.close()
            for ds in datasets:
                ds_name = ds.split()[0]
                out = out.__add__((str(ds_name),))
            out = out.__add__(("Create new dataset...",))
            return out

        dataset = ttk.Combobox(self, values=get_datasets(), justify=tk.CENTER)
        dataset.current(0)
        dataset.pack()

        def next_frame():
            if dataset.current() == len(dataset["values"]) - 1:
                controller.show_frame("DatasetCreation")
            else:
                global datasetName
                datasetName = dataset.get()
                for i in datasets:
                    if i.split()[0] == datasetName:
                        global datasetDirectory
                        datasetDirectory = i.split()[1]
                controller.show_frame("ChooseEpochs")

        tk.Button(self, text='Next', command=next_frame).pack()

        def show_start():
            dataset.current(0)
            controller.show_frame("StartPage")

        tk.Button(self, text="Back", command=show_start).pack()


class DatasetCreation(tk.Frame):
    def __init__(self, parent, controller):
        self.pathToNewSet = ""
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Name the dataset').pack()
        name_label = tk.Entry(self)
        name_label.pack()
        dir_label = tk.Label(self)

        def open_filedialog():
            self.pathToNewSet = askdirectory()
            dir_label['text'] = "Current path: " + self.pathToNewSet
            dir_label.pack()

        def create_start_train(name):
            if self.pathToNewSet != "" and name != "":
                print("WIP", self.pathToNewSet, name)
                global datasetName
                global datasetDirectory
                datasetName = name
                datasetDirectory = self.pathToNewSet
                datasets_file = open("Resources/datasets.txt", 'a')
                datasets_file.write('\n' + datasetName + " " + datasetDirectory)
                datasets_file.close()
                controller.show_frame("ChooseEpochs")

        tk.Button(self, text="Choose the photos folder", command=open_filedialog).pack()
        tk.Button(self, text="Save and start to train", command=lambda: create_start_train(name_label.get())).pack()


class ChooseEpochs(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Select the number of epochs:').pack()
        inp = tk.Entry(self)
        inp.pack()

        def set_epochs(epochCount):
            if epochCount.isdigit():
                print("WIP")
                global epochCounter
                epochCounter = int(epochCount)
                inp.delete(0, 'end')
                controller.show_frame("Training")

        tk.Button(self, text="Generate", command=lambda: set_epochs(inp.get())).pack()


class Training(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        cur_ep = tk.Label(self, text='Current epoch: 0/' + str(epochCounter))

        def start():
            cur_ep.pack()
            train.main(epochCounter, datasetDirectory)
            controller.show_frame("TrainingEnd")

        tk.Button(self, text="Start training", command=start).pack()


class TrainingEnd(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Button(self, text="Return to menu", command=lambda: controller.show_frame("StartPage")).pack()
        tk.Button(self, text="Repeat training", command=lambda: controller.show_frame("ChooseEpochs")).pack()


def get_classes():
    out = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            out.append(obj)
    return out


datasets = []
datasetName = ""
datasetDirectory = ""
epochCounter = 0
active_pages = get_classes()
