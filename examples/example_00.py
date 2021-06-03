from neurobiba import *
from random import random 

#Создаем веса
weights = create_weights([2, 1]) 

#Учим нейросеть определять что первое число больше второго
for i in range(1000): 
    input_layer = [random(),random()]
    output_lauer = [int(input_layer[0]>input_layer[1])]
    weights = training(input_layer, output_lauer, weights) 

#Тестируем обученную нейросеть
correct = 0
for i in range(100):
    input_layer = [random(),random()]
    result = round(feed_forward(input_layer, weights)[0])
    if result == int(input_layer[0]>input_layer[1]): 
        correct += 1

#Результат
print('Нейросеть ответила верно в ', correct, ' случаях из 100')
    
