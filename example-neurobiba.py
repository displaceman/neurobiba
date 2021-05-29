from neurobiba import *
from random import random 

syn = create_weights([2, 1]) #Создаем веса

#Учим нейросеть определять что первое число больше второго
for i in range(1000): 
    inp = [random(),random()]
    syn = training(inp, int(inp[0]>inp[1]), syn) 

#Тестируем обученную нейросеть
t = 0
for i in range(100):
    inp = [random(),random()]
    re = round(result(inp, syn)[0])
    if re == int(inp[0]>inp[1]): t+=1

print('Нейросеть ответила верно в ', t, ' случаях из 100')
    
