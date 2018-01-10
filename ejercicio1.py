import random
import numpy as np
import math

weights = np.matrix([[random.uniform(0.1, 0.3) for j in range(4)] for i in range(11)])

# El último termino, que es igual a 1 es el bias
input_values = [((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0)), 
                ((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 1)), 
                ((0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 1, 0)), 
                ((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1), (0, 0, 1, 1)), 
                ((0, 0, 0, 0, 1, 0 ,0 ,0 ,0 ,0, 1), (0, 1, 0, 0)), 
                ((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), (0, 1, 0, 1)), 
                ((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1), (0, 1, 1, 0)), 
                ((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1), (0, 1, 1, 1)), 
                ((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1), (1, 0, 0, 0)), 
                ((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1), (1, 0, 0, 1))]

def sum_of_products(inputs, weights):
    return np.dot(inputs, weights) # Hace el producto punto entre dos matrices

def logistic_function(net):
    out = []
    for i in range(net.size): # da el tamaño de la matriz (cantidad de elementos)
        out.append(1 / (1 + math.exp(-net.item(i))))
    return out


while True:
    error_count = 0
    for inputs, target in input_values:
        net = sum_of_products(inputs, weights)
        out = logistic_function(net)
        print(out)
        break
    break
            

