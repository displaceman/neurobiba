# neurobiba

![visitors](https://visitor-badge.laobi.icu/badge?page_id=displaceman.neurobiba)

other languages:

- [EN](./README.md)

небольшая библиотека для создания простых нейросетей.

очень простая в использовании!

установка:

```
pip install neurobiba
```

смотрите [примеры](./examples)

![example_01](./examples/example_01.PNG)

# как использовать

попробуем создать простую нейросеть, которая будет определять, больше ли первое число второго.

1. создаём веса. у нашей нейросети будет два входа и один выход.

```python
weights = Weights([2, 1]) # 2 входа и 1 выход
```

2. генерируем входные данные и правильные ответы, а затем обучаем на них нейросеть.

```python
for i in range(10000): # тренируем 10000 раз
    a, b = random(), random() # 'a' и 'b' - случайные числа
    output = int(a > b) # если 'a' > 'b' то ответ 1, иначе 0
    weights.train([a, b], [output]) # тренируем
```

3. радуемся обученной нейросети!

```python
result = weights.feed_forward([0.1, 0.3])[0]
# результат будет близок к нулю, а это значит, что 0.1 < 0.3
```
