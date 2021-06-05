from neurobiba import *

import random
import matplotlib.pyplot as plt

neural_net = Weights([1, 50, 1], bias = True)

x_axis = []
dataset = []
result = []
dimension = 20

def example_function(x):
    x = x+0.5
    return (((4*x**3-6*x**2+1)*(x+1)**0.5)/(3-x)+1)*0.5
    
for i in range(1000):
    for u in range(dimension):
        u /= dimension
        neural_net.train([u], [example_function(u)])

for u in range(dimension):
    u /= dimension
    
    x_axis.append(u)
    dataset.append(example_function(u))
    result.append(neural_net.feed_forward([u])[0])

plt.plot(x_axis, result)
plt.plot(x_axis, dataset)
plt.title('result')
plt.show()
