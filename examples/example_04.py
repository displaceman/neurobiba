from neurobiba import Weights
from neurobiba.color import mix3

from PIL import Image, ImageDraw
from random import random

# Константы
IMAGE_SIZE = WIDTH, HEIGHT = 200, 200
DATA_AMOUNT = 16
TRAINING_ITERATIONS = 1000

# Создание весов
weights = Weights([2, 16, 3])

# Генерация датасета
input_neurons = [[random(), random()] for _ in range(DATA_AMOUNT)]
output_neurons = [[i%3==0, i%3==1, i%3==2] for i in range(DATA_AMOUNT)]

# Процесс обучения
for _ in range(TRAINING_ITERATIONS):
    for i in range(len(input_neurons)):
        weights.train(input_neurons[i], output_neurons[i])

# Создание картинки
image = Image.new('RGB', IMAGE_SIZE, 'white')   
draw = ImageDraw.Draw(image)

# Отрисовка поля
for x in range(WIDTH):
    for y in range(HEIGHT):
        brightness = weights.feed_forward([x/WIDTH, y/HEIGHT])
        draw.point((x, y), mix3(brightness))

# Отрисовка точек
for ind, i in enumerate(input_neurons):
    x, y = int(i[0]*WIDTH), int(i[1]*HEIGHT)
    draw.ellipse((x-5, y-5, x+5, y+5), 
        fill = mix3(output_neurons[ind]), 
        outline = (0, 0, 0))

# Вывод изображения
image.show()
