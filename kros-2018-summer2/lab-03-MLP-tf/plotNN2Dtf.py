import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data
x_data = np.loadtxt('./data/dataX.txt', ndmin=2)
y_data = np.loadtxt('./data/dataY.txt', ndmin=2)
datanum, inputdim = x_data.shape
datanum, outputdim = y_data.shape
x_data = np.column_stack((x_data, np.ones([datanum, 1])))
hiddennum1 = 20
hiddennum2 = 20

# *******************************************************
# Data plotting
# *******************************************************
C1 = np.where(y_data == 0)[0]
C2 = np.where(y_data == 1)[0]
C = (C1, C2)
colors = ("red", "green")

for c, color in zip(C, colors):
   plt.scatter(x_data[c, 0], x_data[c, 1], alpha=1.0, c=color)

# *******************************************************
#                Model building
# *******************************************************
# weight
W1 = tf.Variable(tf.random_uniform([hiddennum1,inputdim+1],-0.1,0.1))
W2 = tf.Variable(tf.random_uniform([hiddennum2,hiddennum1],-0.1,0.1))
W3 = tf.Variable(tf.random_uniform([outputdim,hiddennum2],-0.1,0.1))
X = tf.placeholder(tf.float32, shape=(None, inputdim + 1), name="input")
Y = tf.placeholder(tf.float32, shape=(None, outputdim), name="output")

# Model building
v1 = tf.matmul(W1,tf.transpose(X))      # v1 = W1*X'    (hiddennum1 x datanum)
y1 = tf.nn.sigmoid(v1)                   # y1 = sig(v1)    (hiddennum1 x datanum)
v2 = tf.matmul(W2,y1)                    # v2 = W2*y1    (hiddennum2 x datanum)
y2 = tf.nn.sigmoid(v2)                 # y2 = sig(v2) (hiddennum2 x datanum)
v3 = tf.matmul(W3,y2)                    # v3 = W3*y2    (outputdim x datanum)
Yhat = tf.nn.sigmoid(v3)                 # Yhat = sig(v3) (outputdim x datanum)
Yhat = tf.transpose(Yhat)                # Yhat = datanum x outputdim

# Session opening
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './result/model.ckpt')

# *******************************************************
#                  Contour plotting
# *******************************************************
step = 0.025
x_axis = np.arange(0.0, 1.0 + step, step)
y_axis = np.arange(0.0, 1.0 + step, step)
X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis)
Z_mesh = np.zeros(X_mesh.shape)

for x in range(X_mesh.shape[0]):
   for y in range(Y_mesh.shape[0]):
       input = np.array([X_mesh[x][y], Y_mesh[x][y], 1], ndmin=2)
       Z_mesh[x][y] = sess.run(Yhat, feed_dict={X:input})

plt.contour(x_axis, y_axis, Z_mesh, (0.5), linewidths=3)
plt.show()
