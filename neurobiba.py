from numpy import (exp, random, array, dot)
from pickle import (dump, load)

def sigmoid_derivative(x, alpha): 
    return (x *(1-x))*alpha 


def sigmoid(x): 
    return 1/(1+exp(-x))
    
    
def create_weights(l_size): 
    syn = []
    for i in range(len(l_size)-1):
        syn.append(2*random.random((l_size[i], l_size[i+1])) - 1)
    return syn


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


def result(inp, syn): 
    l = [array([inp])]
    d = len(syn)
    
    for i in range(d):
        l.append(sigmoid(dot(l[-1],syn[i])))

    return l[-1][0]


def reverse(inp, syn):
    synr = list(reversed(syn))
    for ind, i in enumerate(synr):
        synr[ind] = synr[ind].T
        
    l = [array([inp])]
    d = len(synr)

    for i in range(d):
        l.append(sigmoid(dot(l[-1],synr[i])))
    return l[-1][0]

def download_syn(file_name = 'syn'):
    try: 
        with open(f'{file_name}.dat','rb') as file:
            return pickle.load(file)
    except:
        print('no file with saved weights')

def save_syn(syn, file_name = 'syn'):
    with open(f'{file_name}.dat','wb') as file:
        pickle.dump(syn, file)
    print('file saved')


