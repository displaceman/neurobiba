from PIL import Image, ImageDraw
from neurobiba import *
from random import random

#Константы
IMAGE_SIZE = WIDTH, HEIGHT = 500, 500
DATA_AMOUNT = 8
TRAINING_ITERATIONS = 5000

#Создание весов
weights = create_weights([3,16,1])

#Генерация датасета
input_neurons = [[random(), random(), 1] for _ in range(DATA_AMOUNT)]
output_neurons = [i%2 for i in range(DATA_AMOUNT)]

#Процесс обучения
for _ in range(TRAINING_ITERATIONS):
    for i in range(len(input_neurons)):
        weights = training(input_neurons[i], output_neurons[i], weights)

#Создание картинки
image = Image.new('RGB', IMAGE_SIZE, 'white')   
draw = ImageDraw.Draw(image)

#Отрисовка поля
for x in range(WIDTH):
    for y in range(HEIGHT):
        brightness = int(round(feed_forward([x/WIDTH, y/HEIGHT, 1], weights)[0])*255)
        color = tuple([brightness] * 3)
        draw.point((x, y), color)

#Отрисовка точек
for ind, i in enumerate(input_neurons):
    x, y = int(i[0]*WIDTH), int(i[1]*HEIGHT)
    brightness = int(output_neurons[ind]*255)
    rect = (x-5, y-5, x+5, y+5)
    fill = tuple([brightness] * 3)
    outline = tuple(map(lambda a: 255-a, fill))
    draw.ellipse(rect, fill = fill, outline = outline)

#Вывод изображения
image.show()
