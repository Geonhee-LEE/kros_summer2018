import numpy as np
import matplotlib.pyplot as plt
import NN

X= np.loadtxt('./data/dataX.txt',ndmin=2)
Y= np.loadtxt('./data/dataY.txt',ndmin=2)
datanum,inputdim=X.shape
X=np.column_stack((X,np.ones([datanum,1])))
D = Y
outputdim=1

epochnum = 50000;
ERR = np.zeros([epochnum,1]);
hiddennum = 20

W1 = 2*np.random.random((hiddennum,inputdim+1)) - 1
W2 = 2*np.random.random((outputdim,hiddennum)) - 1
        
for epoch in range(epochnum):
    W1, W2 = NN.Backprop(W1, W2, X, D)
    Yhat = NN.NN(X,W1,W2)
    ERR[epoch] = 1/2*np.matmul((D-Yhat).T,(D-Yhat))/datanum
    if epoch%100 == 0:
       print("epoch = %d, Err=%f" % (epoch,ERR[epoch]))

plt.plot(ERR, 'r')
plt.show()
np.savetxt('./result/dataW1.txt', W1)
np.savetxt('./result/dataW2.txt', W2)



