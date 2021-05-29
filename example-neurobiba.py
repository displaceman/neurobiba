from neurobiba import *
from random import random 

weights = create_weights([2, 1]) #Создаем веса

#Учим нейросеть определять что первое число больше второго
for i in range(1000): 
    input_layer = [random(),random()]
    weights = training(input_layer, [int(input_layer[0]>input_layer[1])], weights) 

#Тестируем обученную нейросеть
t = 0
for i in range(100):
    input_layer = [random(),random()]
    result= round(feed_forward(input_layer, weights)[0])
    if result== int(input_layer[0]>input_layer[1]): t+=1

print('Нейросеть ответила верно в ', t, ' случаях из 100')
    
