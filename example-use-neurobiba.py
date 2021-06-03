from PIL import Image, ImageDraw
from neurobiba import *
from random import random as r

#Создание картинки
image = Image.new('RGB', (500, 500), 'white')   
width, height = image.size
draw = ImageDraw.Draw(image)

#Создание весов
weights = create_weights([3,16,1])

#Создание датасета
input_neurons = []
output_neurons = []
for i in range(8):
    input_neurons.append([r(), r(), 1])
    output_neurons.append(i%2)

#Процесс обучения
for i in range(3000):
    for ind in range(len(input_neurons)):
        weights = training(input_neurons[ind], output_neurons[ind], weights)

#Отрисовка поля
for x in range(width):
    for y in range(height):
        c = int(round(feed_forward([x/width, y/height, 1], weights)[0])*255)
        draw.point((x, y), (c,c,c))

#Отрисовка точек
for ind, i in enumerate(input_neurons):
    x, y = int(i[0]*width), int(i[1]*height)
    c = int(output_neurons[ind]*255)
    draw.ellipse((x-5, y-5, x+5, y+5), fill = (c,c,c), outline = tuple(map(lambda a: 255-a, (c,c,c))))

#Вывод изображения
image.show()
