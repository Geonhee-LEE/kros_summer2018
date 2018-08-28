from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

X1 = mnist.train.images
Y1 = mnist.train.labels
X2 = mnist.test.images
Y2 = mnist.test.labels

print(X1.shape, Y1.shape)
print(X2.shape, Y2.shape)

i = 5
print(Y1[i])
image = X1[i].reshape((28,28))
plt.imshow(image, cmap='gray')
plt.waitforbuttonpress()