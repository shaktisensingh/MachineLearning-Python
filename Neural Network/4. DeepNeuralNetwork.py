# Deep Neural Network. Network size contains number of nodes per layer.

import numpy as np
import matplotlib.pyplot as plot

def main():
    alpha = 0.01  # Learing rate
    iterations = 5000  # number of iterations
    costs = np.zeros((1,2))  # Initialization of cost array
    fileName = 'XOR.txt'
    X, Y = readData(fileName)
    
    # Number of nodes per layer. For Ex: input, hidden and oytput layes have 2, 3, 1 nodes respectively.
    networkSize = np.array([2, 3, 1])  
    Ws = initializeWeights(networkSize)

    for i in range(iterations):
        As = forwardProp(X, Ws)
        dWs = backProp(As, Y, Ws)
        Ws = update_weights(networkSize, Ws, dWs, alpha)

        cost = costFun(As['A'+str(networkSize.shape[0]-1)], Y)
        costs = np.append(costs, [[i, cost]], axis = 0)

    plotCost(costs[1:-1,:])
    print(Ws)  # Final tuned paramenters.
    return

def plotCost(costs):
    plot.scatter(costs[:, 0], costs[:, 1])
    plot.xlabel('Number of Iteration')
    plot.ylabel('Cost')
    plot.show(block = False)
    return

def update_weights(networkSize, Ws, dWs, alpha):
    L = len(networkSize)
    for i in reversed(range(1, L)):
        Ws['W'+str(i)] -= alpha * dWs['dW'+str(i)]
        Ws['b'+str(i)] -= alpha * dWs['db'+str(i)]
    return Ws

def backProp(As, Y, Ws):
    dAs = {}  # dictionary for derivative of error wrt activation function for all the layers
    dWs = {}  # dictionary for weights for all the layers
    L = len(As)
    AL = As['A'+str(L-1)]  # Final output of the network
    dAs['dA'+str(L-1)] = -np.divide(Y, AL) + np.divide((1-Y), (1-AL))  # Derivative of error wrt to activation funtion of output layer
    for i in reversed(range(1, L)):
        dWs['dW'+str(i)], dWs['db'+str(i)], dAs['dA'+str(i-1)] = backwardActivation(dAs['dA'+str(i)], As['A'+str(i)], As['A'+str(i-1)], Ws['W'+str(i)], Ws['b'+str(i)])
    return dWs

def backwardLinear(dZ, A):
    dW=np.dot(dZ, A.T)
    return dW

def backwardActivation(dA, A, A_prev, W, b):
    dZ = dA * sigmoidPrime(A)
    dW = backwardLinear(dZ, A_prev)
    db = np.sum(dZ, keepdims = True)
    dA = np.dot(W.T, dZ)
    return dW, db, dA

def sigmoidPrime(A):
    return A*(1-A)

def costFun(A, Y):
    m = Y.shape[1]
    cost = np.sum(np.multiply(-Y, np.log(A))-np.multiply((1-Y), np.log(1-A)), axis=1, keepdims=True)/m
    return cost

def forwardProp(X, weights):    
    As = {}  # initializing dictionary
    As['A0'] = X
    L = len(weights)//2  # returns integer
    for i in range(1, L+1):
        temp = forwardActivation(As['A'+str(i-1)], weights['W'+str(i)], weights['b'+str(i)])
        As['A'+str(i)] = temp
    return As

def forwardActivation(A, W, b):
    Z = forwardLinear(A, W, b)
    A = sigmoid(Z)
    return A

def forwardLinear(A, W, b):
    Z = np.dot(W, A) + b
    return Z

def initializeWeights(networkSize):
    L = len(networkSize)
    weights = {}
    for i in range(1, L):
        weights['W' + str(i)] = np.random.randn(networkSize[i], networkSize[i - 1])
        weights['b' + str(i)] = np.zeros((networkSize[i], 1))  # Bias
    return weights
                              
def sigmoid(x):
    return 1/(1+np.exp(-x))

def readData(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    x = data[:, 0:2].T
    y = data[:, 2][:, None].T
    return x, y
                              
if __name__ == '__main__':
    main()
