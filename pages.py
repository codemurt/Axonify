import tkinter as tk
import random


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the start page", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        tk.Button(self, text="Go to Page One", command=lambda: controller.show_frame("PageOne")).pack()
        tk.Button(self, text="Go to Page Two", command=lambda: controller.show_frame("PageTwo")).pack()
        tk.Button(self, text="Go to Page Three", command=lambda: controller.show_frame("PageThree")).pack()
        tk.Button(self, text="Go to Page Four", command=lambda: controller.show_frame("PageFour")).pack()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 1", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage")).pack()
        tk.Button(self, text="Go to Page Two", command=lambda: controller.show_frame("PageTwo")).pack()
        tk.Button(self, text="Go to Page Three", command=lambda: controller.show_frame("PageThree")).pack()


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 2", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage")).pack()


class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 3", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage")).pack()


class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        title = tk.Label(self, text="Введите сид")
        input_label = tk.Entry(self, fg="green", bg="lightgray", borderwidth=5, state=tk.DISABLED)
        input_label.insert(0, "")
        output_label = tk.Label(self, text="Тут будет изображение")

        def applyClick():
            text = "Здесь будет картинка"
            nums = input_label.get()
            if not nums == "":
                text += " c номером " + nums
            output_label['text'] = text

        def numButtonClick(num):
            input_label['state'] = tk.NORMAL
            input_label.insert(tk.END, str(num))
            input_label['state'] = tk.DISABLED

        def clearInput():
            input_label['state'] = tk.NORMAL
            input_label.delete(0, tk.END)
            input_label['state'] = tk.DISABLED

        def clearLast():
            input_label['state'] = tk.NORMAL
            input_label.delete(len(input_label.get()) - 1, tk.END)
            input_label['state'] = tk.DISABLED

        def randomNum():
            clearInput()
            input_label['state'] = tk.NORMAL
            input_label.insert(0, str(random.randint(1, 10000)))
            input_label['state'] = tk.DISABLED

        apply_button = tk.Button(self, text="Сгенерировать", padx=50, pady=20, bg="red", activebackground="green",
                                 command=applyClick)

        title.grid(row=0, column=0, columnspan=11)
        input_label.grid(row=1, column=0, columnspan=11)
        apply_button.grid(row=4, column=0, columnspan=11)
        output_label.grid(row=5, column=0, columnspan=11)

        tk.Button(self, text=1, command=lambda: numButtonClick(1)).grid(row=2, column=0)
        tk.Button(self, text=2, command=lambda: numButtonClick(2)).grid(row=2, column=1)
        tk.Button(self, text=3, command=lambda: numButtonClick(3)).grid(row=2, column=2)
        tk.Button(self, text=4, command=lambda: numButtonClick(4)).grid(row=2, column=3)
        tk.Button(self, text=5, command=lambda: numButtonClick(5)).grid(row=2, column=4)
        tk.Button(self, text=6, command=lambda: numButtonClick(6)).grid(row=2, column=5)
        tk.Button(self, text=7, command=lambda: numButtonClick(7)).grid(row=2, column=6)
        tk.Button(self, text=8, command=lambda: numButtonClick(8)).grid(row=2, column=7)
        tk.Button(self, text=9, command=lambda: numButtonClick(9)).grid(row=2, column=8)
        tk.Button(self, text=0, command=lambda: numButtonClick(0)).grid(row=2, column=9)
        tk.Button(self, text="Clear last", command=clearLast).grid(row=2, column=10)
        tk.Button(self, text="X", command=clearInput, width=15).grid(row=3, column=0, columnspan=5)
        tk.Button(self, text="Random", command=randomNum, width=24).grid(row=3, column=5, columnspan=6)
        tk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))\
            .grid(row=6, column=0, columnspan=11)


active_pages = (StartPage, PageOne, PageTwo, PageThree, PageFour)
