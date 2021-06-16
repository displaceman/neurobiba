import neurobiba as nb

from PIL import Image, ImageDraw
from random import random


#Константы
IMAGE_SIZE = WIDTH, HEIGHT = 200, 200
DATA_AMOUNT = 8
TRAINING_ITERATIONS = 5000


#Создание весов
weights = nb.Weights([2,8,1])

#Генерация датасета
input_neurons = [[random(), random()] for _ in range(DATA_AMOUNT)]
output_neurons = [i%2 for i in range(DATA_AMOUNT)]

#Процесс обучения
for _ in range(TRAINING_ITERATIONS):
    for i in range(len(input_neurons)):
        weights.train(input_neurons[i], output_neurons[i])

#Создание картинки
image = Image.new('RGB', IMAGE_SIZE, 'white')   
draw = ImageDraw.Draw(image)

#Отрисовка поля
for x in range(WIDTH):
    for y in range(HEIGHT):
        brightness = int(round(weights.feed_forward([x/WIDTH, y/HEIGHT])[0])*255)
        color = tuple([brightness] * 3)
        draw.point((x, y), color)

#Отрисовка точек
for ind, i in enumerate(input_neurons):
    x, y = int(i[0]*WIDTH), int(i[1]*HEIGHT)
    rect = (x-5, y-5, x+5, y+5)
    fill = tuple([int(output_neurons[ind]*255)] * 3)
    draw.ellipse(rect, fill = fill, outline = tuple(map(lambda a: 255-a, fill)))

#Вывод изображения
image.show()
