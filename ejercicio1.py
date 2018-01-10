import random
import numpy as np
import math


learning_rate = 0.5

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

def get_output_error(output, target):
    errors = []
    for i in range(len(output)):
        errors.append(math.pow(1/2*(target[i] - output[i]), 2))
    return errors

def error_total(errors):
    return sum(errors)

def delta_rule(out_j, target):
    return -(target - out_j) * out_j * (1 - out_j) # (d E_total)/(d w_j)

def modify_weight(weight, xj, delta, learning_rate):
    return weight - learning_rate * delta * xj



while True:
    error_count = 0
    for inputs, target in input_values:
        net = sum_of_products(inputs, weights)
        out = logistic_function(net)
        errors = get_output_error(out, target)
        error = error_total(errors)
        # BackPropagation
        weights_t = np.transpose(weights)
        for i in range(len(out)):
            delta = delta_rule(out[i], target[i])
            
                
        break
    break
            

