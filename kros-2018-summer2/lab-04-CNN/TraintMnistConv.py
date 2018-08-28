import numpy as np
import matplotlib.pyplot as plt
import mnist
import NN


X,D = mnist.load_train_datasets("./MNIST_data", one_hot=True)
X = np.array(X).T
X = X/255                            # uint8 [0,1,2, ...255] -> float [0,1] normalize
X = X.reshape((28,28,1,-1))          # H x W x ChanNum x DataNum
(xrow, xcol,_, datanum) = X.shape   # H x W x ChanNum x DataNum
datanum,outputdim=D.shape

# ********************************************************
# Draw MNIST
# ********************************************************
fig = plt.figure()
ims = np.random.randint(datanum, size=15)

for i in range(15):
    subplot = fig.add_subplot(3,5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title("%d" %np.argmax(D[ims[i]]))
    subplot.imshow(X[:,:,0,ims[i]],cmap='gray')

plt.show()



# ********************************************************
#                           Training
# ********************************************************
epochnum = 100
ERR = np.zeros([epochnum, 1])
learningRate = 0.01

# Weight Initialize
#        W1 (5x5x1x3)
L1Size = 5
L1ChaNum = 1
L1FilterNum = 3
W1 = 1e-3*(2 * np.random.random_sample((L1Size, L1Size, L1ChaNum, L1FilterNum)) - 1)

#       W2 (5x5x3x20)
L2Size = 5
L2ChaNum = L1FilterNum
L2FilterNum = 20
W2 = 1e-3*(2 * np.random.random_sample((L2Size, L2Size, L2ChaNum, L2FilterNum)) - 1)


#  2000 --> 300
L5HidNum = 300
W5 = 1e-1*(2 * np.random.random_sample((L5HidNum, 2000)) - 1)

# 300 --> 10
Wo = 1e-1*(2 * np.random.random_sample((outputdim, L5HidNum)) - 1)



for epoch in range(epochnum):
    # Train
    W1, W2, W5, Wo = NN.BackPropCNN(W1, W2, W5, Wo, X, D, learningRate)

    # Evaluation
    Yhat = NN.CNN(W1, W2, W5, Wo, X)
    ERR[epoch] = 1 / 2 * np.trace(np.matmul((D - Yhat).T, (D - Yhat))) / datanum
    if epoch % 1 == 0:
        print("epoch = %d, Err=%f" % (epoch, ERR[epoch]))

plt.plot(ERR, 'r')
plt.show()

np.savez('./result/nnW.npz', W1=W1, W2=W2, W5=W5, Wo=Wo)
# For W1 = W1, the left one is the index name for the element, the right one is the variable name)
#C=np.load('./result/nnW.npz')
#print(C.keys())
#print(C['W1'])
#print(C['W5'])






