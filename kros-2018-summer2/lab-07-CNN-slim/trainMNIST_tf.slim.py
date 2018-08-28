import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

x_data = mnist.train.images
x_data = x_data.reshape((-1, 28, 28))
x_data = np.expand_dims(x_data, axis=-1)
y_data = mnist.train.labels

n_data, h, w, n_channels = x_data.shape
n_data, n_classes = y_data.shape

# ********************************************************
#               Draw MNIST
# ********************************************************
fig = plt.figure()
idxs = np.random.randint(n_data, size=15)

for i in range(15):
    subplot = fig.add_subplot(3, 5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title("%d" % np.argmax(y_data[idxs[i]]))
    subplot.imshow(x_data[idxs[i], :, :, 0], cmap='gray')

plt.show()


# ********************************************************
#               Training
# ********************************************************

# Training Parameters

epochs = 10
batch_size = 256
batches = int(n_data / batch_size)
learning_rate = 0.001
display_step = 10

# *************************************************************
#               Model building
# Convolutional layers then typically apply a ReLU activation function
# to the output to introduce nonlinearities into the model.
# *************************************************************


X = tf.placeholder(tf.float32, shape=(None, w, h, 1), name="input")   # (None means 'any number of rows')
Y = tf.placeholder(tf.float32, shape=(None, n_classes), name="output")

with slim.arg_scope(
    [slim.conv2d, slim.fully_connected],
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_uniform_initializer(-0.01, 0.01)):
    # Layer 2 (Convolution), W1 (5x5x1x3):  L1 (28x28x1) --> L2 (24x24x3)
    y1 = slim.conv2d(X, 3, 5, padding='valid', scope='conv_1')

    # Layer 3 (Convolution), W2 (5x5x3x20): L2 (24x24x3) --> L3 (20x20x20)
    y2 = slim.conv2d(y1, 20, 5, padding='valid', scope='conv_2')

    # Layer 4 (Pooling): L3 (20x20x20) --> L4 (10x10x20)
    y3 = slim.avg_pool2d(y2, 2, scope='avg_pool_3')

    # Layer 5 (Flatten): L4 (10x10x20) --> L5 (2000)
    y4 = tf.layers.flatten(y3)

    # Layer 6 (Fully-connected), W3 (2000x300): L5 (2000) --> L6 (300)
    y5 =  slim.fully_connected(y4, 300, scope='fc_5')

    # Layer 7 (Fully-connected for Output), Wo (300x10): L6 (300) --> L7 (10)
    v = slim.fully_connected(y5, n_classes, activation_fn=None, scope='fc_6')

# Cost function (Cross entropy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=v), axis=0)

# Optimization option
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = slim.learning.create_train_op(cost, optimizer)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(v, 1), tf.argmax(Y, 1)), tf.float32))

# summary scalar for loss
tf.summary.scalar('cost', cost)

# merge all summary ops into a single op
summary_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# write summary to disk for Tensorboard visualization
writer = tf.summary.FileWriter('./result/', sess.graph)

for epoch in range(epochs):
    for batch in range(batches):

        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape((-1, 28, 28))
        x_batch = np.expand_dims(x_batch, axis=-1)
        _ = sess.run([train_op, cost], feed_dict={X:x_batch, Y:y_batch})

        step = (epoch * batches) + batch + 1
        if (step % display_step) == 0:
            costval, accval, summary = sess.run([cost, accuracy, summary_op], feed_dict={X: x_batch, Y: y_batch})
            print("step: %d, epoch: %d, cost: %f, accuracy: %.4f" % (step, epoch + 1, costval, accval))
            # write summary events to disk
            writer.add_summary(summary, step)

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './result/model.ckpt')


