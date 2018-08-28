import numpy as np
import matplotlib.pyplot as plt
import SVM


def plotSVM(X,Y,SV_X, SV_Y, SV_a,b_bound,sig):

    C1 = np.where(Y== -1)[0]
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
            input = np.array([X_mesh[x,y], Y_mesh[x,y]],ndmin=2)
            Z_mesh[x,y] =  SVM.evalSVM(input,SV_X, SV_Y, SV_a,b_bound,sig)

    plt.contour(x_axis, y_axis, Z_mesh,(0.0))
    #contour(X,Y,Z,V)
    #draw contour lines at the values specified in sequence V,
    #which must be in increasing order.

    plt.show()


if __name__ == '__main__':

    X= np.loadtxt('./data/dataX.txt',ndmin=2)
    Y= np.loadtxt('./data/dataY.txt',ndmin=2)
    datanum,dim=X.shape

    W = np.load('./result/SVM.npz')
    SV_X = W['SV_X']
    SV_Y = W['SV_Y']
    SV_a = W['SV_a']
    b_bound = W['b_bound']

    sig = 0.1
    plotSVM(X, Y, SV_X, SV_Y, SV_a, b_bound, sig)
   
