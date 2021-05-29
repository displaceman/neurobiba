from numpy import (exp, random, array, dot)
from pickle import (dump, load)

"""
Функция sigmoid_derivative используется другими функциями библиотеки.
"""
def sigmoid_derivative(x, alpha): 
    return (x *(1-x))*alpha 

"""
Функция sigmoid используется другими функциями библиотеки в качестве функции активации.
"""
def sigmoid(x): 
    return 1/(1+exp(-x))


"""
Функция create_weights нужна для создания весов. 

l_size это список слоев с количеством их нейронов.

Пример использования функции:
weights = create_weights([3,10,10,2])
Здесь три нейрона на входном слое, 
два промежуточных слоя по 10 нейронов и 2 нейрона на выходе.
"""
def create_weights(l_size): 
    weights = []
    for i in range(len(l_size)-1):
        weights.append(2*random.random((l_size[i], l_size[i+1])) - 1)
    return weights


"""
Функция training нужна для обучения нейросети 
методом обратного распространения ошибки.
Возвращает измененные веса. 
Ее нужно вызывать в цикле столько раз сколько потребуется для обучения.

Пример использования функции:
for i in range(1000):
    weights = training(input_layer, correct_output, weights)
    
input_layer это список нейронов входного слоя. 
Они должны находится в диапазоне от 0 до 1.
Например: input_layer = [1, 0, 0.7]

correct_output это список правильных выходов для заданных входов.
Они должны находится в диапазоне от 0 до 1.
Например: correct_output = [0.5, 1]

weights это веса нейросети, которые вы создали ранее

alpha это коэфициент скорости обучения. 
Его оптимальное значение меняется в зависимости от задачи.
"""
def training(input_layer, correct_output, weights, alpha = 0.9):
    l = [array([input_layer])]
    d = len(weights)
    
    for i in range(d):
        l.append(sigmoid(dot(l[-1],weights[i])))
       
    l_error = []
    l_delta = []
 
    l_error.append(correct_output - l[-1] )
    l_delta.append(l_error[-1] * sigmoid_derivative(l[-1], alpha) )

    for i in range(d-1):
        l_error.append(l_delta[i].dot(weights[d-1-i].T))
        l_delta.append(l_error[-1] * sigmoid_derivative(l[d-1-i], alpha))
    
    for ind, i in enumerate(weights):
        weights[ind] += l[ind].T.dot(l_delta[-1-ind])
        
    return weights


"""
Функция feed_forward нужна для получения ответа нейросети.
Возвращает список выходных нейронов.

Пример использования:
r = result(input_layer, weights)

input_layer это список входных нейронов.

weights это веса нейросети, которые вы создали ранее.
"""
def feed_forward(input_layer, weights): 
    l = [array([input_layer])]
    d = len(weights)
    
    for i in range(d):
        l.append(sigmoid(dot(l[-1],weights[i])))

    return l[-1][0]


"""
Функция feed_reverse работает так же как result,
но проводит сигнал через нейросеть в обратном направлении.
В нее в качестве входного слоя подают то что ранее считалось выходным слоем 


Пример использования:
r = reverse(input_layer, weights)
"""
def feed_reverse(input_layer, weights):
    weightsr = list(reversed(weights))
    for ind, i in enumerate(weightsr):
        weightsr[ind] = weightsr[ind].T
        
    l = [array([input_layer])]
    d = len(weightsr)

    for i in range(d):
        l.append(sigmoid(dot(l[-1],weightsr[i])))
    return l[-1][0]


"""
Функция download_weights нужна для загрузки весов из файла

Пример использования:
weights = download_weights()

В качестве аргумента file_name можно указать имя файла .dat без указания формата.
"""
def download_weights(file_name = 'weights'):
    try: 
        with open(f'{file_name}.dat','rb') as file:
            return pickle.load(file)
    except:
        print('no file with saved weights')


"""
Функция save_weights нужна для сохранения весов в файл.

Пример использования:
save_weights(weights)

В качестве аргумента file_name можно указать имя файла .dat без указания формата.
"""
def save_weights(weights, file_name = 'weights'):
    with open(f'{file_name}.dat','wb') as file:
        pickle.dump(weights, file)
    print('file saved')


