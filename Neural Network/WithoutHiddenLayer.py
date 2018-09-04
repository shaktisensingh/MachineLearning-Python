# This network takes two input paramenters and outputs one node and contains no hidden layers. Examples: AND, OR gate

import numpy as np

def main():
    inputLayer_size = 2
    outputLayer_size = 1
    x, y = readData()
    m, n = x.shape

    # Random Initianization of paramenters w and b
    w = np.random.randn(outputLayer_size * inputLayer_size).reshape(outputLayer_size, inputLayer_size)
    b = 0
    
    for i in np.arange(5000):       # Number of iterations = 5000
        # Forward propagation
        z = np.dot(w, x) + b        # Linera activation
        a = sigmoid(z)                # Sigmoid activation (Non-Linear)
        
        # Backward propagation
        dz = a - y                                                   # Partial derivative of error wrt Linear activation
        dw = np.dot(dz, x.T) / m                               # Partial derivative wrt to weights w
        db = np.sum(a - y, axis = 1, keepdims = True)   # Partial derivative wrt bias

        # Paramenter updation
        w -= 0.01 * dw              # 0.01 is learing rate Alpha
        b -= 0.01 * db
        
    print('Bias',b[0,0])
    print('Weight 1',w[0,0])
    print('Weight 2',w[0,1])
    return

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def readData():
    data = np.loadtxt('AND_OR.txt', delimiter = ',')
    y = data[:,2][:,None]
    x = data[:,0:2]
    return x.T, y.T

if __name__=="__main__":
    main()
