from neurobiba import Weights
from neurobiba.color import mix3, randcolor

from PIL import Image, ImageDraw
from random import random

# Константы
IMAGE_SIZE = WIDTH, HEIGHT = 500, 500
DATA_AMOUNT = 16
TRAINING_ITERATIONS = 10000

iteration = 0
while True:
    iteration += 1

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

    # Генерация цветов
    colors = [randcolor(),randcolor(),randcolor()]

    # Отрисовка поля
    for x in range(WIDTH):
        for y in range(HEIGHT):
            brightness = weights.feed_forward([x/WIDTH, y/HEIGHT])
            draw.point((x, y), mix(brightness, colors))

    # Отрисовка точек
    for ind, i in enumerate(input_neurons):
        x, y = int(i[0]*WIDTH), int(i[1]*HEIGHT)

        draw.ellipse((x-5, y-5, x+5, y+5), 
            fill = mix(output_neurons[ind], colors), 
            outline = (0, 0, 0))

    # Вывод изображения
    #image.show()
    image.save(str(iteration)+".png", 'png')