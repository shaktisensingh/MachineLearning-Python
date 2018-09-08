# Deep Neural Network. Network size contains number of nodes per layer.

import numpy as np

def main():
    fileName = 'XOR.txt'
    X, Y = readData(fileName)
    networkSize = np.array([2, 3, 1])
    Ws = initializeWeights(networkSize)
##    print('Ws',Ws)
    As = forwardProp(X, Ws)
    print('As',As)
    cost=costFun(As['A'+str(networkSize.shape[0])],Y)
    print('Cost',cost)
    print('---------------------')
    dWs=backProp(As,Y,Ws)
    print(dWs)
    return

def backProp(As,Y,Ws):
    L=len(As)
    print('L',L)
    dAs={}
    AL=As['A'+str(L)]
    dAs['dA'+str(L)]=-np.divide(Y,AL)+np.divide((1-Y),(1-AL))
    dWs={}
    for i in reversed(range(2,L+1)):
        print('Back Prop Loop = ',i)
        dWs['dW'+str(i)],dAs['dA'+str(i-1)]=backwardActivation(dAs['dA'+str(i)],As['A'+str(i)],As['A'+str(i-1)],Ws['W'+str(i-1)])
        print('dWs',dWs)
    return dWs

def backwardLinear(dZ,A):
    print('dZ',dZ)
    print('backwardLinera A', A.T)
    dW=np.dot(dZ,A.T)
    print('dW',dW)
    return dW

def backwardActivation(dA,A,A_prev,W):
    dZ=dA*sigmoidPrime(A)
    dA=np.dot(A_prev,dZ.T)
    print('dZ',dZ.shape)
    print('A_prev',A_prev.shape)
    print(dA.shape)
    dW=backwardLinear(dZ,A_prev)
    return dW,dA

def sigmoidPrime(A):
    A_prime=A*(1-A)
    print('A_prime',A_prime)
    return A_prime

def costFun(A,Y):
    m=Y.shape[1]
    cost=np.sum(np.multiply(-Y,np.log(A))-np.multiply((1-Y),np.log(1-A)),axis=1,keepdims=True)/m
    return cost

def forwardProp(X,weights):    
    As = {}
    As['A1'] = X
    L = len(weights)//2
    print('L',L)
    for i in range(1,L+1):
        A=forwardActivation(As['A'+str(i)],weights['W'+str(i)],weights['b'+str(i)])
        A_prev=A
        As['A'+str(i+1)]=A
    return As

def forwardActivation(A,W,b):
    Z=forwardLinear(A,W,b)
    A=sigmoid(Z)
    return A

def forwardLinear(A,W,b):
    Z=np.dot(W,A)+b
    return Z

def initializeWeights(networkSize):
    L = len(networkSize)
    weights = {}
    for i in range(1, L):
        weights['W' + str(i)] = np.random.randn(networkSize[i], networkSize[i - 1]) * 0.01
        weights['b' + str(i)] = np.zeros((networkSize[i], 1))
    return weights
                              
def sigmoid(x):
    return 1/(1+np.exp(-x))

def readData(fileName):
    data=np.loadtxt(fileName,delimiter=",")
    x=data[:,0:2].T
    y=data[:,2][:,None].T
    return x,y
                              
if __name__=='__main__':
    main()
