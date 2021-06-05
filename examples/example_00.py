import neurobiba as nb

from random import random 

#Создаем веса
weights = nb.Weights([2, 1]) 

#Учим нейросеть определять, больше ли первое число второго
for _ in range(1000): 
    first, second = random(), random()
    result = int(first > second)
    weights.training([first, second], [result]) 

#Тестируем обученную нейросеть
correct = 0
for _ in range(100):
    first, second = random(), random()
    result = round(weights.feed_forward([first, second])[0])
    if result == int(first > second): 
        correct += 1

#Результат
print(f'Нейросеть ответила верно в {correct} случаях из 100')
