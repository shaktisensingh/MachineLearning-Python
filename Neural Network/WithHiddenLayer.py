# Simple Neural Network with one input, Hidden and output Layer. Example: XOR Gate

import numpy as np
import matplotlib.pyplot as plot

def main():
    alpha = 0.1  # Learning rate
    file_name = 'XOR.txt'
    costs = np.zeros((1,2))  # Initilization of costs array
    size_input, size_hidden, size_output = defineLayers()
    x, y = readData(file_name)
    weights = initializeWeights(size_input, size_hidden, size_output)
    for i in np.arange(1000):
        result_f = forwardProp(weights, x)
        
        cost = calculateCost(result_f['a2'], y)
        costs = np.append(costs, [[i, cost]], axis = 0)
        
        gradients = backProp(weights, result_f, x, y)
        weights = update_weights(weights, gradients, alpha)
        
    plotCost(costs)  # plots cost vs iterations

    # Prints final weights
    print('W1')
    for i in np.arange(size_input):
        for j in np.arange(size_hidden):
            p = weights["w1"]
            print(p[j, i])
            
    print('\n B1')
    for i in np.arange(size_hidden):
        p = weights["b1"]
        print(p[i, 0])
        
    print('\n W2') 
    for i in np.arange(size_hidden):
        for j in np.arange(size_output):
            p = weights["w2"]
            print(p[j, i])
    print('\n B2')
    for i in np.arange(size_output):
        p = weights["b2"]
        print(p[i, 0])
    return

def update_weights(weights, gradients, alpha):
    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]
    dw2 = gradients["dw2"]
    db2 = gradients["db2"]
    dw1 = gradients["dw1"]
    db1 = gradients["db1"]
    
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w1 -= alpha * dw1
    b1 -= alpha * db2
    params = {"w1":w1, "b1":b1, "w2":w2, "b2":b2}
    return params

def backProp(weights, result_f, x, y):
    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]
    z1 = result_f["z1"]
    a1 = result_f["a1"]
    z2 = result_f["z2"]
    a2 = result_f["a2"]
    
    dz2 = a2 - y  # derivative of error wrt linear activation of output layer
    dw2 = np.dot(dz2, a1.T)  # # derivative of error wrt weights of output layer
    db2 = np.sum(dz2, axis = 1, keepdims = True)  # derivative of error wrt bias of output layer
    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)  # derivative of error wrt linear activation of hidden layer
    dw1 = np.dot(dz1, x.T)  # derivative of error wrt weights of hidden layer 
    db1 = np.sum(dz1, axis = 1, keepdims = True)  # derivative of error wrt bias of hidden layer
    params = {"dw2":dw2, "db2":db2, "dw1":dw1, "db1":db1}
    return params

def plotCost(costs):
    plot.scatter(costs[:, 0], costs[:, 1])
    plot.xlabel('Iterantion')
    plot.ylabel('Cost')
    plot.show(block = False)
    return

def calculateCost(a, y):
    cost = np.sum(np.multiply(-y, np.log(a)) - np.multiply((1 - y),np.log(1 - a)), axis = 1, keepdims = True)/y.shape[1]
    return cost

def forwardProp(weights,x):
    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]
    
    z1 = np.dot(w1, x) + b1  # Linear activation on hidden layer
    a1 = sigmoid(z1)  # Non-Linear (Sigmoid) activaton on hidden layer
    z2 = np.dot(w2, a1) + b2  # Linear activation on output layer
    a2 = sigmoid(z2)  # Non-Linear activation (Sigmoid) on output layer
    params = {"z1":z1, "a1":a1, "z2":z2, "a2":a2}
    return params

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def initializeWeights(size_input,size_hidden,size_output):
    # Initialization of weights between input and hidden layer
    w1 = np.random.randn(size_hidden, size_input)
    b1 = np.zeros((size_hidden, 1))  # Bias

    # Initialization of weights between hidden and output layer
    w2 = np.random.randn(size_output, size_hidden)
    b2 = np.zeros((size_output, 1))  # Bias

    # Filling all the weights in dictionary
    weights = {"w1":w1, "b1":b1, "w2":w2, "b2":b2}
    return weights

def defineLayers():
    size_input = 2  # Number of nodes in input layer
    size_hidden = 3  # Number of nodes in hidden layer
    size_output = 1  # number of nodes in final layer
    return size_input,size_hidden,size_output

def readData(file_name):
    data = np.loadtxt(file_name, delimiter = ',')
    x = data[:,0:2].T
    y = data[:,2][:,None].T
    return x, y

if __name__ == "__main__":
    main()
