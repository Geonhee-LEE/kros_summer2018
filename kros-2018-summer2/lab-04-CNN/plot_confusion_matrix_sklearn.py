import numpy as np
import matplotlib.pyplot as plt
import mnist
import NN
import sklearn.metrics as skl


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    X,D = mnist.load_test_datasets("./MNIST_data", one_hot=True)
    X = np.array(X).T
    X = X / 255  # uint8 [0,1,2, ...255] -> float [0,1] normalize
    X = X.reshape((28,28,1,-1))          # H x W x ChanNum x DataNum
    (xrow, xcol,_, datanum) = X.shape   # H x W x ChanNum x DataNum
    (_,outputdim) =D.shape

    W = np.load('./result/nnW.npz')
    W1 = W['W1']
    W2 = W['W2']
    W5 = W['W5']
    Wo = W['Wo']

    Yhat = NN.CNN(W1, W2, W5, Wo, X)


    ylable = np.argmax(D,axis=1)
    yhatlable = np.argmax(Yhat,axis=1)

    # Compute confusion matrix
    cnf_matrix = skl.confusion_matrix(ylable, yhatlable)
    np.set_printoptions(precision=2)


    # Plot non-normalized confusion matrix
    class_names = ['0','1','2','3','4','5','6','7','8','9']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


