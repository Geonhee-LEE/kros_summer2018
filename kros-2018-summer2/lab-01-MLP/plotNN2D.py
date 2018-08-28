import numpy as np
import matplotlib.pyplot as plt
import NN


def plotNN2D(X,Y,W1,W2):

    C1 = np.where(Y== 0)[0]
    C2 = np.where(Y== 1)[0]
    C = (C1,C2)
    colors = ("red", "green")

    for c,color in zip(C, colors):
        plt.scatter(X[c,0], X[c,1], alpha=1.0, c=color)

    step = 0.025
    x_axis = np.arange(0.0, 1.0+step, step)
    y_axis = np.arange(0.0, 1.0+step, step)
    X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis)
    Z_mesh = np.zeros(X_mesh.shape)

    for x in range(X_mesh.shape[0]):
        for y in range(Y_mesh.shape[0]):
            input = np.array([X_mesh[x][y], Y_mesh[x][y], 1],ndmin=2)
            Z_mesh[x][y] =  NN.NN(input,W1, W2)
            #input = input.T
            #v = np.matmul(W,input)
            #Z_mesh[x][y] =  Sigmoid(v);
            
    plt.contour(x_axis, y_axis, Z_mesh,(0.5))
    #contour(X,Y,Z,V)
    #draw contour lines at the values specified in sequence V,
    #which must be in increasing order.

    plt.show()


if __name__ == '__main__':

    X= np.loadtxt('./data/dataX.txt',ndmin=2)
    Y= np.loadtxt('./data/dataY.txt',ndmin=2)
    W1= np.loadtxt('./result/dataW1.txt',ndmin=2)
    W2= np.loadtxt('./result/dataW2.txt',ndmin=2)
    datanum,dim=X.shape
    plotNN2D(X,Y,W1,W2)
   
