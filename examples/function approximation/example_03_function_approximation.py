from neurobiba import *

import random
import matplotlib.pyplot as plt
import math

neural_net = Weights([1, 50, 1], bias = True)

x_axis = []
dataset = []
result = []
dimension = 20

def example_function(x):
    return (math.sin(x*10)+1)*0.5
    
for i in range(10000):
    for u in range(dimension):
        u /= dimension
        neural_net.train([u], [example_function(u)])

for u in range(dimension):
    u /= dimension
    x_axis.append(u)
    dataset.append(example_function(u))
    result.append(neural_net.feed_forward([u])[0])

plt.title('function approximation')
plt.plot(x_axis, dataset, label = 'example function')
plt.plot(x_axis, result, label = 'neural network response')
plt.legend()
plt.show()
