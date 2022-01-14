# DCGAN + RealESRGAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aQsXD-HvieYQts-X1nqPcZU1uSGNDh5i?usp=sharing)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

**Название проекта:** Нейросети как инструмент формирования научно ориентированных абстракций

**Название команды:** *Клюква*

**Формат системы:** Desktop application or jupyter notebook

**Цель:**

Тренировка нейросетей для создания абстрактных изображений, используемых научным изданием.

**Описание:**

Наш продукт нацелен на создание новых изображений для научного издания на основе нейронных сетей.

**Целевая аудитория:**

Главные редакторы научных журналов. А также те, кому необходимо создание уникальных изображений на своём датасете.

**Основное преимущество:**

Нашим главным преимуществом является графический интерфейс (GUI), поэтому людям не погружённым в программирование будет не сложно разобраться в нашем продукте, и использовать его для решения своих задач. К тому же существует версия для Google Colab, для тех, кому не хочется устанавливать Python, CUDA, и их зависимости или отсутсвует мощное железо. 

**Стек технологий:**

1. Язык программирования Python
2. PyTorch — фреймворк машинного обучения для языка Python
3. Tkinter — для создания GUI
4. OpenCV — библиотека алгоритмов компьютерного зрения, обработки изображений.

**Работа пользователя с системой:**


**Основные требования к ПО для использования:**


**Порядок установки:**

```
1. git clone https://github.com/codemurt/DCGAN.git

2. cd DCGAN

3. ...
```

**Структура приложения:**

`/interface.py` - точка входа для GUI 

`/pages.py` - файл, содержащий логику и описание окон для tkinter 

`/train.py` - основной файл, в котором происходит вся работа нейронной сети, её обучение, создание датасетов и генерация изображений 

`/train_nb.py` - обучение нейронной сети для jupyter notebooks 

`/requirements.txt` - файл, содержащий все зависимости нашего приложения 

`/RealESRGAN/` - директория, в которой лежить библиотека для улучшения изображений при генерации. 
