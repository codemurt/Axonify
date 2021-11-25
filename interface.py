import random
from tkinter import *


def createWindow():
    root = Tk()
    root.title("Генератор изображений")
    title = Label(root, text="Введите сид")
    input_label = Entry(root, fg="green", bg="lightgray", borderwidth=5, state=DISABLED)
    input_label.insert(0, "")
    output_label = Label(root, text="Тут будет изображение")

    def applyClick():
        text = "Здесь будет картинка"
        nums = input_label.get()
        if not nums == "":
            text += " c номером " + nums
        output_label['text'] = text

    def numButtonClick(num):
        input_label['state'] = NORMAL
        input_label.insert(END, str(num))
        input_label['state'] = DISABLED

    def clearInput():
        input_label['state'] = NORMAL
        input_label.delete(0, END)
        input_label['state'] = DISABLED

    def clearLast():
        input_label['state'] = NORMAL
        input_label.delete(len(input_label.get())-1, END)
        input_label['state'] = DISABLED

    def randomNum():
        clearInput()
        input_label['state'] = NORMAL
        input_label.insert(0, str(random.randint(1, 10000)))
        input_label['state'] = DISABLED

    apply_button = Button(root, text="Сгенерировать", padx=50, pady=20, bg="red", activebackground="green",
                          command=applyClick)

    title.grid(row=0, column=0, columnspan=11)
    input_label.grid(row=1, column=0, columnspan=11)
    apply_button.grid(row=4, column=0, columnspan=11)
    output_label.grid(row=5, column=0, columnspan=11)

    Button(root, text=1, command=lambda: numButtonClick(1)).grid(row=2, column=0)
    Button(root, text=2, command=lambda: numButtonClick(2)).grid(row=2, column=1)
    Button(root, text=3, command=lambda: numButtonClick(3)).grid(row=2, column=2)
    Button(root, text=4, command=lambda: numButtonClick(4)).grid(row=2, column=3)
    Button(root, text=5, command=lambda: numButtonClick(5)).grid(row=2, column=4)
    Button(root, text=6, command=lambda: numButtonClick(6)).grid(row=2, column=5)
    Button(root, text=7, command=lambda: numButtonClick(7)).grid(row=2, column=6)
    Button(root, text=8, command=lambda: numButtonClick(8)).grid(row=2, column=7)
    Button(root, text=9, command=lambda: numButtonClick(9)).grid(row=2, column=8)
    Button(root, text=0, command=lambda: numButtonClick(0)).grid(row=2, column=9)
    Button(root, text="Clear last", command=clearLast).grid(row=2, column=10)
    Button(root, text="X", command=clearInput, width=15).grid(row=3, column=0, columnspan=5)
    Button(root, text="Random", command=randomNum, width=24).grid(row=3, column=5, columnspan=6)

    root.mainloop()


if __name__ == '__main__':
    createWindow()
