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

В качестве агрумента принимает список размеров слоев нейросети. 
Возможное слоев от 2 до бесконечности.

Пример использования функции:
syn = create_weights([3,10,10,2])
Здесь три нейрона на входном слое, 
два промежуточных слоя по 10 нейронов 
и 2 нейрона на выходе.
"""
def create_weights(l_size): 
    syn = []
    for i in range(len(l_size)-1):
        syn.append(2*random.random((l_size[i], l_size[i+1])) - 1)
    return syn


"""
Функция training нужна для обучения нейросети 
методом обратного распространения ошибки.
Возвращает измененные веса. 
Ее нужно вызывать в цикле столько раз сколько потребуется для обучения.

Пример использования функции:
for i in range(1000):
    syn = training(inp, correct_output, syn)
    
inp это список нейронов входного слоя. 
Они должны находится в диапазоне от 0 до 1.
Например: inp = [1, 0, 0.7]

correct_output это список правильных выходов для заданных входов.
Они должны находится в диапазоне от 0 до 1.
Например: correct_output = [0.5, 1]

syn это веса нейросети, которые вы создали ранее

alpha это коэфициент скорости обучения. 
Его оптимальное значение меняется в зависимости от задачи.
"""
def training(inp, correct_output, syn, alpha = 0.9):
    l = [array([inp])]
    d = len(syn)
    
    for i in range(d):
        l.append(sigmoid(dot(l[-1],syn[i])))
       
    l_error = []
    l_delta = []
 
    l_error.append(correct_output - l[-1] )
    l_delta.append(l_error[-1] * sigmoid_derivative(l[-1], alpha) )

    for i in range(d-1):
        l_error.append(l_delta[i].dot(syn[d-1-i].T))
        l_delta.append(l_error[-1] * sigmoid_derivative(l[d-1-i], alpha))
    
    for ind, i in enumerate(syn):
        syn[ind] += l[ind].T.dot(l_delta[-1-ind])
        
    return syn


"""
Функция result нужна для получения ответа нейросети.
Возвращает список выходных нейронов.

Пример использования:
r = result(inp, syn)

inp это список входных нейронов.

syn это веса нейросети, которые вы создали ранее.
"""
def result(inp, syn): 
    l = [array([inp])]
    d = len(syn)
    
    for i in range(d):
        l.append(sigmoid(dot(l[-1],syn[i])))

    return l[-1][0]


"""
Функция reverse работает так же как result,
но проводит сигнал через нейросеть в обратном направлении.
В нее в качестве входного слоя подают то что ранее считалось выходным слоем 


Пример использования:
r = reverse(inp, syn)
"""
def reverse(inp, syn):
    synr = list(reversed(syn))
    for ind, i in enumerate(synr):
        synr[ind] = synr[ind].T
        
    l = [array([inp])]
    d = len(synr)

    for i in range(d):
        l.append(sigmoid(dot(l[-1],synr[i])))
    return l[-1][0]


"""
Функция download_syn нужна для загрузки весов из файла

Пример использования:
syn = download_syn()

В качестве аргумента file_name можно указать имя файла .dat без указания формата.
"""
def download_syn(file_name = 'syn'):
    try: 
        with open(f'{file_name}.dat','rb') as file:
            return pickle.load(file)
    except:
        print('no file with saved weights')


"""
Функция save_syn нужна для сохранения весов в файл.

Пример использования:
save_syn(syn)

В качестве аргумента file_name можно указать имя файла .dat без указания формата.
"""
def save_syn(syn, file_name = 'syn'):
    with open(f'{file_name}.dat','wb') as file:
        pickle.dump(syn, file)
    print('file saved')
