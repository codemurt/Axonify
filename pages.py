import inspect
import json
import os
import random
import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font as tkfont
from tkinter.filedialog import askdirectory

import train


def get_datasets():
    out = ()
    global datasets

    for _, dirs, _ in os.walk("Datasets"):
        for dr in dirs:
            out = out.__add__((str(dr),))
            with open("Datasets/" + dr + "/vars.json") as f:
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
        label = tk.Label(self, text="DCGAN",
                         font=tkfont.Font(family='Helvetica', size=25, weight="bold", slant="italic"))
        label.pack(side="top", fill="x", pady=10)

        tk.Button(self, text="Generate", height=4, width=20,
                  command=lambda: controller.show_frame("GeneratorPage")).pack(side="top", pady=15)
        tk.Button(self, text="Training", height=4, width=20,
                  command=lambda: controller.show_frame("DatasetSelection")).pack(side="top")


class GeneratorPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self, text="Select dataset:", font=controller.title_font)
        label1.pack(side="top", pady=5)
        dataset_selector = ttk.Combobox(self, values=get_datasets(), justify=tk.CENTER, width=15)
        dataset_selector.pack(side='top')
        label2 = tk.Label(self, text="Print number of images to generate:", font=controller.title_font)
        label2.pack(side="top", pady=10)
        epoch_count = tk.Entry(self, width=8, justify='center')
        epoch_count.pack(side='top')
        label3 = tk.Label(self, text="Print seed or leave blank to random:", font=controller.title_font)
        label3.pack(side="top", pady=10)
        seed = tk.Entry(self, width=8, justify='center')
        seed.pack(side="top")

        def generate_pics(imgCount, in_seed, show_images):
            if not in_seed.isdigit():
                in_seed = random.randint(0, 9999)
                print(f"Random seed: {in_seed}")
            if imgCount.isdigit() and not dataset_selector.current() == -1:
                epoch_count.delete(0, 'end')
                seed.delete(0, 'end')
                global datasetName
                datasetName = dataset_selector.get()
                global datasetDirectory
                for i in datasets:
                    if len(i) > 3 and i.split()[0] == datasetName:
                        datasetDirectory = i.split()[1]
                        break
                print(f"Generating {imgCount} image(s) with DSet {datasetName} and seed {in_seed}...")
                train.generate(datasetName, int(in_seed), show_images, imgCount)
                print("Generation finished.")
                controller.show_frame("GenerationEnd")

        def show_start():
            epoch_count.delete(0, 'end')
            controller.show_frame("StartPage")

        image_show = tk.BooleanVar()
        image_show.set(False)
        tk.Checkbutton(self, text="Show generated images", variable=image_show, onvalue=1, offvalue=0) \
            .pack(side="top", pady=5)
        tk.Button(self, text="Generate", command=lambda: generate_pics(epoch_count.get(), seed.get(), image_show.get()),
                  width=10).pack(side="top", pady=5)
        tk.Button(self, text="Back", command=show_start, width=10).pack(side="top", pady=5)


class GenerationEnd(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Ready! Your pictures are saved in the "ResultImages" folder.',
                 font=controller.title_font).pack()
        tk.Button(self, text='Return to menu', command=lambda: controller.show_frame("StartPage")).pack()


class DatasetSelection(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Select dataset:', font=controller.title_font).pack(side='top', pady=5)

        datasets_selector = ttk.Combobox(self, values=get_datasets_with_new(), justify=tk.CENTER, width=20)
        datasets_selector.current(0)
        datasets_selector.pack(side='top', pady=10)

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
                        print(f"Chosen dataset: {datasetName}, directory: {datasetDirectory}")
                        controller.show_frame("ChooseEpochs")
                        break

        tk.Button(self, text='Next', command=next_frame, width=15).pack(side='top')

        def show_start():
            datasets_selector.current(0)
            controller.show_frame("StartPage")

        tk.Button(self, text="Back", command=show_start, width=15).pack(side='top', pady=10)


class DatasetCreation(tk.Frame):
    def __init__(self, parent, controller):
        self.pathToNewSet = ""
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Name the dataset', font=controller.title_font).pack(side='top', pady=10)
        name_label = tk.Entry(self, width=24, justify='center')
        name_label.pack(side='top', pady=10)
        dir_label = tk.Label(self)

        def open_filedialog():
            self.pathToNewSet = askdirectory()
            dir_label['text'] = "Current path: " + self.pathToNewSet
            dir_label.pack(side='top', pady=5)
            print(f"Directory: {self.pathToNewSet}")

        def create_start_train(name):
            if self.pathToNewSet != "" and name != "":
                name_label.delete(0, 'end')

                global datasetName
                global datasetDirectory
                datasetName = "".join(name.split())
                datasetDirectory = self.pathToNewSet
                vars_directory = "Datasets/" + datasetName

                if not os.path.exists(vars_directory):
                    os.makedirs(vars_directory)

                if not os.path.exists(vars_directory + "/vars.json"):
                    with open(vars_directory + "/vars.json", "w") as f:
                        f.write('{"directory": "' + datasetDirectory + '", "image_iterator": 0, "epochs": 0}')
                print(f"Created new dataset. Name: {datasetName}, directory: {datasetDirectory}.")
                controller.show_frame("ChooseEpochs")

        def back():
            controller.show_frame("DatasetSelection")

        tk.Button(self, text="Choose the photos folder", command=open_filedialog, width=20).pack(side='top')
        tk.Button(self, text="Save", command=lambda: create_start_train(name_label.get()), width=10) \
            .pack(side='top', pady=10)
        tk.Button(self, text="Back", command=back, width=10).pack(side='top')


class ChooseEpochs(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Print the number of epochs:', font=controller.title_font).pack(side='top', pady=10)
        inp1 = tk.Entry(self, width=10, justify='center')
        inp1.pack(side='top', pady=10)

        def set_values(epochCount):
            if epochCount.isdigit():
                epochCounter = int(epochCount)
                inp1.delete(0, 'end')
                print(f"Starting training on DSet {datasetName}"
                      f" located in {datasetDirectory}, epoch number: {epochCounter}")
                train.train(epochCounter, "Datasets/" + datasetName, datasetDirectory, datasetName)
                controller.show_frame("TrainingEnd")

        def back():
            controller.show_frame("DatasetSelection")

        tk.Button(self, text="Train", command=lambda: set_values(inp1.get()), width=10) \
            .pack(side='top')
        tk.Button(self, text="Back", command=back, width=10).pack(side='top', pady=10)


class TrainingEnd(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text='Training finished.', font=controller.title_font) \
            .pack(side="top", pady=10)
        tk.Button(self, text="Return to menu", width=20, command=lambda: controller.show_frame("StartPage")) \
            .pack(side="top", pady=10)
        tk.Button(self, text="Repeat training", width=20, command=lambda: controller.show_frame("ChooseEpochs")) \
            .pack(side="top")


def get_pages():
    return [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(obj)]


datasets = []
datasetName = ""
datasetDirectory = ""
