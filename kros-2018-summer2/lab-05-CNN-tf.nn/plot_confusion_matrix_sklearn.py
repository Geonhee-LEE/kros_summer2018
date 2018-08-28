import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as skl
from tensorflow.examples.tutorials.mnist import input_data


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
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    x_data = mnist.test.images
    y_data = mnist.test.labels
    x_data = x_data.reshape((-1, 28, 28))
    x_data = np.expand_dims(x_data, axis=-1)

    (n_data, h, w, n_channels) = x_data.shape
    (_, n_classes) = y_data.shape


    # *************************************************************
    #               Model building
    # *************************************************************

    # Layer 1 (Input): L1 (28x28x1)
    X = tf.placeholder(tf.float32, shape=(None, w, h, 1), name="input")   # (None means 'any number of rows')
    Y = tf.placeholder(tf.float32, shape=(None, n_classes), name="output")

    # Layer 2 (Convolution), W1 (5x5x1x3):  L1 (28x28x1) --> L2 (24x24x3)
    W1 = tf.Variable(tf.random_uniform([5, 5, 1, 3], -0.01, 0.01), name="W1")
    v1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
    y1 = tf.nn.relu(v1)

    # Layer 3 (Convolution), W2 (5x5x3x20): L2 (24x24x3) --> L3 (20x20x20)
    W2 = tf.Variable(tf.random_uniform([5, 5, 3, 20], -0.01, 0.01), name="W2")
    v2 = tf.nn.conv2d(y1, W2, strides=[1, 1, 1, 1], padding='VALID')
    y2 = tf.nn.relu(v2)

    # Layer 4 (Pooling): L3 (20x20x20) --> L4 (10x10x20)
    y3 = tf.nn.avg_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 5 (Flatten): L4 (10x10x20) --> L5 (2000)
    y4 = tf.reshape(y3, [-1, 10*10*20])

    # Layer 6 (Fully-connected), W3 (2000x300): L5 (2000) --> L6 (300)
    W3 = tf.Variable(tf.random_uniform([2000, 300], -0.01, 0.01), name="W3")
    v5 = tf.matmul(y4, W3)
    y5 = tf.nn.relu(v5)

    # Layer 7 (Fully-connected for Output), Wo (300x10): L6 (300) --> L7 (10)
    Wo = tf.Variable(tf.random_uniform([300, n_classes], -0.01, 0.01), name="Wo")
    Yhat = tf.matmul(y5, Wo)

    # Session opening
    # saver = tf.train.Saver()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, './result/model.ckpt')

    pred, target = sess.run([Yhat, Y], feed_dict={X: x_data, Y: y_data})

    ylable = np.argmax(target,axis=1)
    yhatlable = np.argmax(pred,axis=1)

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


