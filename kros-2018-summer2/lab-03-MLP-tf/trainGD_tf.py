import tensorflow as tf
import numpy as np


x_data = np.loadtxt('./data/dataX.txt',ndmin=2)
y_data = np.loadtxt('./data/dataY.txt',ndmin=2)
datanum,inputdim=x_data.shape
datanum,outputdim=y_data.shape
x_data = np.column_stack((x_data,np.ones([datanum,1])))
hiddennum1 = 20
hiddennum2 = 20

# placeholder (None means 'any number of rows')
X = tf.placeholder(tf.float32,shape=(None,inputdim+1),name="input")
Y = tf.placeholder(tf.float32,shape=(None,outputdim),name="output")

# weight
W1 = tf.Variable(tf.random_uniform([hiddennum1,inputdim+1],-0.1,0.1))
W2 = tf.Variable(tf.random_uniform([hiddennum2,hiddennum1],-0.1,0.1))
W3 = tf.Variable(tf.random_uniform([outputdim,hiddennum2],-0.1,0.1))

# Model building
v1 = tf.matmul(W1,tf.transpose(X))      # v1 = W1*X'    (hiddennum1 x datanum)
y1 = tf.nn.sigmoid(v1)                   # y1 = sig(v1)    (hiddennum1 x datanum)
v2 = tf.matmul(W2,y1)                    # v2 = W2*y1    (hiddennum2 x datanum)
y2 = tf.nn.sigmoid(v2)                 # y2 = sig(v2) (hiddennum2 x datanum)
v3 = tf.matmul(W3,y2)                    # v3 = W3*y2    (outputdim x datanum)
Yhat = tf.nn.sigmoid(v3)                 # Yhat = sig(v3) (outputdim x datanum)
Yhat = tf.transpose(Yhat)                # Yhat = datanum x outputdim

#Cost function
cost = tf.reduce_mean(tf.reduce_sum((Y-Yhat)**2,axis=1))


# Optimization option
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
sess = tf.Session()
sess.run(init)

for step in range(5000):
    sess.run(train_op,feed_dict={X:x_data, Y:y_data})

    if step % 1000 == 0:
        costval = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print("%d: cost = %f"%(step, costval))

# print some predictions compared to target values
save_path = saver.save(sess, './result/model.ckpt')
pred, target = sess.run([Yhat, Y], feed_dict={X:x_data, Y:y_data})
print(pred[:10],target[:10])


