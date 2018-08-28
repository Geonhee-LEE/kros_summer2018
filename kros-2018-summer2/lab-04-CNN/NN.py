import numpy as np
import scipy.signal as sp

#******************************************************
#                 NN forward pass
#            Convolutional neural network
#******************************************************

def CNN(W1, W2, W5, Wo, X):
    (_,_,_,datanum) = X.shape      # data number
    (classnum,_) = Wo.shape   # output number

    Yhat = np.zeros((datanum, classnum))

    for k in range(datanum):
        x = X[:,:,:,k]                       # Input,           28x28
        v1 = Conv(x, W1)              # Convolution,  24x24x3
        y1 = ReLu(v1)                 # ReLu 24x24x3
        v2 = Conv(y1, W2)            # Convolution,  20x20x20
        y2 = ReLu(v2)                 # ReLu 20x20x20
        y3 = Pool(y2)                  # Pooling,      10x10x20
        y4 = np.reshape(y3, (-1, 1),order='F')       #  10x10x20 = 2000
        v5 = np.matmul(W5, y4)                    # ReLu,             300
        y5 = ReLu(v5)                 #
        v = np.matmul(Wo, y5)                    # Softmax,          10x1
        y = softmax(v)                      # softmax
        Yhat[k, :] = y.T

    return Yhat





#******************************************************
#                 NN backward pass
#******************************************************

def BackPropCNN(W1, W2, W5, Wo, X, D, alpha):
    # alpha is a learning rate
    beta = 0.95             # Moment

    momentum1 = np.zeros(W1.shape)
    momentum2 = np.zeros(W2.shape)
    momentum5 = np.zeros(W5.shape)
    momentumo = np.zeros(Wo.shape)

    datanum = D.shape[0]

    bsize = 100
    blist = np.arange(0, datanum, bsize)


    # One epoch loop
    #
    for batch in range(len(blist)):
        dW1 = np.zeros(W1.shape)
        dW2 = np.zeros(W2.shape)
        dW5 = np.zeros(W5.shape)
        dWo = np.zeros(Wo.shape)

        # Mini - batch  loop
        begin = blist[batch]
        for k in range(begin,begin+bsize):

            # Forward pass = inference     #
            x = X[:,:,:,k]   # Input, 28x28
            d = np.array(D[k,:], ndmin=2).T

            v1 = Conv(x, W1)  # Convolution,  24x24x3
            y1 = ReLu(v1)  # ReLu 24x24x3
            v2 = Conv(y1, W2)  # Convolution,  20x20x20
            y2 = ReLu(v2)  # ReLu 20x20x20
            y3 = Pool(y2)  # Pooling,      10x10x20
            y4 = np.reshape(y3, (-1, 1),order='F')  # 10x10x20 = 2000
            v5 = np.matmul(W5, y4)  # ReLu,             300
            y5 = ReLu(v5)  #
            v = np.matmul(Wo, y5)  # Softmax,          10x1
            y = softmax(v)  # softmax


            # Backpropagation for each data
            e = d - y # Output layer(10x1)
            delta = e

            e5 = np.matmul(Wo.T, delta)             # Hidden(ReLU) layer (300x1)
            delta5 = ((v5 > 0)*1)* e5

            e4 = np.matmul(W5.T,delta5)            # Hidden(ReLU) layer (2000x1)

            e3 = np.reshape(e4, y3.shape,order='F')    # Reshape layer 10x10x20

            e2 = np.zeros(y2.shape)          # Pooling layer 20x20x20
            W3 = np.ones(y2.shape) / (2 * 2)
            for c in range(W2.shape[3]):
                e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]

            delta2 = ((y2 > 0) * 1) * e2  # ReLU layer 20x20x20


            # When delta2 is propagated backward to e1,
            # the size of delta2 is increased by zeros padding.
            # When X2 = W2*H2 and size(W2) = [n2, m2],
            # delta2 is augmented by zero padding by [n2-1, m2-1]
            (n2, m2, _, _) = W2.shape       #W2 5x5x3x20
            delta2padding = np.pad(delta2, ((n2-1,n2-1), (m2-1,m2-1), (0,0)),
                                   'constant', constant_values=((0,0),(0,0),(0,0)))
            # 20x20x20 --> 28X28X20
            e1 = np.zeros(y1.shape)

            for c in range(W2.shape[2]):  # Repeat the times of number of channels  4-->3
                W2channelwise = W2[:,:, c,:]  #W2 5x5x3x20,  W2channelwise = 5x5x20, delta2padding = 28X28X20
                e1[:,:, c] = np.squeeze(Conv(delta2padding, np.rot90(W2channelwise, k=2, axes=(0,1))),axis=-1)
            delta1 = ((y1 > 0)*1) * e1  # 24x24x3

            # ******************************************************
            # Weight update
            #  ******************************************************

            # Convolution layer
            # In case of three channels,  X = W1*H1 + W2*H2 + W3*H3
            # THus, W1 = W1 + Delta*H1
            # THus, W2 = W2 + Delta*H2
            # THus, W3 = W3 + Delta*H3
            # That is, the convolution is conducted separately for each channel
            # The same step is repeated for each filter
            delta1_x = np.zeros(W1.shape)          # Convolutional layer
            for f in range(W1.shape[3]):            #  # of filters
                for c in range(W1.shape[2]):        #  # of channels
                    delta1_x[:,:, c, f] = sp.convolve2d(x[:,:,c],
                                            np.rot90(delta1[:,:,f], k=2), 'valid')

            delta2_x = np.zeros(W2.shape)          # Convolutional layer
            for f in range(W2.shape[3]):  # # of filters
                for c in range(W2.shape[2]):  # # of channels
                    delta2_x[:,:, c, f] = sp.convolve2d(y1[:,:,c],
                                            np.rot90(delta2[:,:, f], k=2), 'valid')


            dW1 = dW1 + delta1_x
            dW2 = dW2 + delta2_x
            dW5 = dW5 + np.matmul(delta5, y4.T)
            dWo = dWo + np.matmul(delta, y5.T)

        # Update weights for minibatch
        dW1 = dW1 / bsize
        dW2 = dW2 / bsize
        dW5 = dW5 / bsize
        dWo = dWo / bsize

        momentum1 = alpha * dW1 + beta * momentum1
        W1 = W1 + momentum1

        momentum2 = alpha * dW2 + beta * momentum2
        W2 = W2 + momentum2

        momentum5 = alpha * dW5 + beta * momentum5
        W5 = W5 + momentum5

        momentumo = alpha * dWo + beta * momentumo
        Wo = Wo + momentumo

    return W1, W2, W5, Wo



#******************************************************
#                 NN Convolution layer
#******************************************************

def Conv(x, W):

    if (len(x.shape) < 3):
        x = np.expand_dims(x,axis=-1)
    if (len(W.shape) < 4):
        W = np.expand_dims(W,axis=-1)

    wrow, wcol, numChans, numFilters = W.shape
    xrow, xcol, numChans = x.shape

    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1

    y = np.zeros((yrow, ycol, numFilters))


    for k in range(numFilters):
        filters = W[:, :, :, k]
        filters = np.rot90(filters, k=2, axes=(0,1))

        for c in range(numChans):
            y[:, :, k] = y[:, :, k]+ sp.convolve2d(x[:,:,c], filters[:,:,c], 'valid')

    return y



#******************************************************
#                Pooling layer
#******************************************************

def Pool(x):
#
# 2x2 mean pooling
#
#
    xrow, xcol, numFilters = x.shape
    y = np.zeros((int(xrow/2), int(xcol/2), numFilters))
    for k in range(numFilters):
        filters = np.ones((2,2)) / (2*2)    # for mean filtering
        image = sp.convolve2d(x[:,:, k], filters, 'valid')
        y[:, :, k] = image[::2, ::2]

    return y


# *****************************************************
#                 NN ReLu
# ******************************************************
def ReLu(x):
    #  ReLu = max(0,x)
    return np.maximum(0, x)


#******************************************************
#                 NN Logistic sigmoid
#******************************************************
def Sigmoid(x):
    #return 1.0 / (1.0 + np.exp(-x))
    return np.exp(-np.logaddexp(0, -x))



#******************************************************
#                 NN softmax
#******************************************************

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
