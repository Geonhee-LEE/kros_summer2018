import numpy as np


#******************************************************
#                 NN forward pass
#******************************************************

def NN(X,W1,W2):

    #datanum,dim=X.shape
    # X is row-wise samples

    v1 = np.matmul(W1, X.T)
    v1 = v1.T
    Y1 = Sigmoid(v1)

    v2 = np.matmul(W2, Y1.T)
    v2 = v2.T
    Y = Sigmoid(v2)
    return Y


#******************************************************
#                 NN backward pass
#******************************************************

def Backprop(W1, W2, X, D):

    alpha = 0.05  # 0.3
    datanum, dim = X.shape;

    for k in range(datanum):
        x = np.array(X[k, :], ndmin=2).T
        d = np.array(D[k], ndmin=2)

        # NN forward
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)

        v2 = np.matmul(W2, y1)
        y = Sigmoid(v2)

        # NN backward

        e = d - y
        delta = y *(1 - y) * e
	
        e1 = np.matmul(W2.T , delta)
        delta1 = y1 * (1 - y1) * e1

        dW1 = alpha * np.matmul(delta1, x.T)  # delta rule
        W1 = W1 + dW1

        dW2 = alpha * np.matmul(delta, y1.T)
        W2 = W2 + dW2

    return W1, W2


#******************************************************
#                 NN Logistic sigmoid
#******************************************************
def Sigmoid(x):

    #return 1.0 / (1.0 + np.exp(-x))
    return np.exp(-np.logaddexp(0, -x))
