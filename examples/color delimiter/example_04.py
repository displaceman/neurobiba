from neurobiba import Weights
from neurobiba.color import mix, randcolor

from PIL import Image, ImageDraw
from random import random

# Константы
NUMBER_OF_COLORS = 4
IMAGE_SIZE = WIDTH, HEIGHT = 100, 100
DATA_AMOUNT = 16
TRAINING_ITERATIONS = 10000


iteration = 0
while True:
    iteration += 1

    # Создание весов
    weights = Weights([2, 16, NUMBER_OF_COLORS])

    # Генерация датасета
    input_neurons = [[random(), random()] for _ in range(DATA_AMOUNT)]
    output_neurons = [[i%NUMBER_OF_COLORS==ind for ind in range(NUMBER_OF_COLORS)] for i in range(DATA_AMOUNT)]

    # Процесс обучения
    for _ in range(TRAINING_ITERATIONS):
        for i in range(len(input_neurons)):
            weights.train(input_neurons[i], output_neurons[i])

    # Создание картинки
    image = Image.new('RGB', IMAGE_SIZE, 'white')   
    draw = ImageDraw.Draw(image)

    # Генерация цветов
    colors = [randcolor() for _ in range(NUMBER_OF_COLORS)]

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
    image.show()
    image.save(str(iteration)+".png", 'png')
