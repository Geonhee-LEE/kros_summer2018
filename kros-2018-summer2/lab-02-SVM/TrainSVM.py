import numpy as np
import matplotlib.pyplot as plt
import SVM


X= np.loadtxt('./data/dataX.txt',ndmin=2)
Y= np.loadtxt('./data/dataY.txt',ndmin=2)
datanum,inputdim=X.shape
outputdim=1

C = 10
sig = 0.1
Ker = SVM.rbf_kernel(X,X,sig)

# Optimize by cvxopt
a_train = SVM.quadprogSVM(Ker, Y, C)

epsilon = 1e-4
SV_index = np.nonzero( a_train>epsilon )[0]

SV_X = X[SV_index,:]
SV_Y = Y[SV_index,:]
SV_a = a_train[SV_index,:]

Y_out_temp = np.matmul(Ker[:,SV_index],(SV_a* SV_Y))
b_bound = np.mean(Y - np.matmul(Ker,(a_train*Y)))
Y_out_final = Y_out_temp + b_bound
err_train = np.sum(np.sign(Y_out_final) !=Y) / datanum

np.savez('./result/SVM.npz', SV_X=SV_X, SV_Y=SV_Y,
         SV_a=SV_a, b_bound=b_bound)


