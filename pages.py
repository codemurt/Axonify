import inspect
import json
import os
import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askdirectory


# import train

def get_datasets():
    out = ()
    global datasets

    for _, dirs, _ in os.walk("Resources"):
        for dr in dirs:
            out = out.__add__((str(dr),))
            with open("Resources/" + dr + "/vars.json") as f:
                data_json = json.load(f)
            datasets.append(dr + " " + data_json['directory'])

    return out


def get_datasets_with_new():
    out = get_datasets()
    return out.__add__(("Create new dataset...",))


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
        label = tk.Label(self, text="Select dataset and number of images to generate:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        dataset_selector = ttk.Combobox(self, values=get_datasets(), justify=tk.CENTER)
        dataset_selector.pack()
        epoch_count = tk.Entry(self)
        epoch_count.pack()

        def generate_pics(imgCount):
            if imgCount.isdigit() and not dataset_selector.current() == -1:
                print("WIP")
                #
                # do stuff
                #
                epoch_count.delete(0, 'end')
                controller.show_frame("GenerationEnd")

        def show_start():
            epoch_count.delete(0, 'end')
            controller.show_frame("StartPage")

        tk.Button(self, text="Generate", command=lambda: generate_pics(epoch_count.get())).pack()
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

        datasets_selector = ttk.Combobox(self, values=get_datasets_with_new(), justify=tk.CENTER)
        datasets_selector.current(0)
        datasets_selector.pack()

        def next_frame():
            if datasets_selector.current() == len(datasets_selector["values"]) - 1:
                controller.show_frame("DatasetCreation")
            else:
                global datasetName
                datasetName = datasets_selector.get()
                for i in datasets:
                    if len(i) > 3 and i.split()[0] == datasetName:
                        global datasetDirectory
                        datasetDirectory = i.split()[1]
                controller.show_frame("ChooseEpochs")

        tk.Button(self, text='Next', command=next_frame).pack()

        def show_start():
            datasets_selector.current(0)
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
                name_label.delete(0, 'end')
                print(self.pathToNewSet, name)
                global datasetName
                global datasetDirectory
                datasetName = "".join(name.split())
                datasetDirectory = self.pathToNewSet
                vars_directory = "Resources/" + datasetName

                if not os.path.exists(vars_directory):
                    os.makedirs(vars_directory)

                if not os.path.exists(vars_directory + "/vars.json"):
                    with open(vars_directory + "/vars.json", "w") as f:
                        f.write('{"directory": "' + datasetDirectory + '", "image_iterator": 0, "epochs": 0}')

                controller.show_frame("ChooseEpochs")

        def back():
            controller.show_frame("DatasetSelection")

        tk.Button(self, text="Choose the photos folder", command=open_filedialog).pack()
        tk.Button(self, text="Save", command=lambda: create_start_train(name_label.get())).pack()
        tk.Button(self, text="Back", command=back).pack()


class ChooseEpochs(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Select the number of epochs:').pack()
        inp = tk.Entry(self)
        inp.pack()

        def set_epochs(epochCount):
            if epochCount.isdigit():
                global epochCounter
                epochCounter = int(epochCount)
                inp.delete(0, 'end')
                # train.main(epochCounter, jsonDirectory, datasetDirectory)
                print(epochCounter, datasetDirectory, datasetName)
                controller.show_frame("TrainingEnd")

        def back():
            controller.show_frame("DatasetSelection")

        tk.Button(self, text="Generate", command=lambda: set_epochs(inp.get())).pack()
        tk.Button(self, text="Back", command=back).pack()


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
