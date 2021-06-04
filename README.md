# neurobiba

![visitors](https://visitor-badge.laobi.icu/badge?page_id=displaceman.neurobiba)

small collection of functions for neural networks.

very easy to use!

Installation:

```
pip install neurobiba
```

See [examples](./examples)

![example_01](./examples/example_01.PNG)

# how to use

1. create weights

```python
weights = create_weights([2, 1]) # 2 input neurons and 1 output
```

2. create data, create answer, train

```python
for i in range(1000): # train 1000 times
    a, b = random(), random() # a and b is a random numbers
    output = int(a > b) # if a > b then answer is 1, else 0
    weights = training([a, b], [output], weights) # train
```

3. enjoy

```python
result = feed_forward([0.1, 0.3], weights)[0] # result is close to 0
```
