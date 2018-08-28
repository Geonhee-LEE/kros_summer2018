import numpy as np
import tensorflow as tf

INPUT_DIM = 4
ACTION_DIM = 2

class Network(object): # Policy and Value Network
    def __init__(self, device):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(device):
                self.create_placeholder()
                self.create_network()
                self.create_train_op()
                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer()) 

    def create_placeholder(self):
        self.state_t = tf.placeholder(tf.float32, [None, INPUT_DIM], name='state_t') # states
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr') # accumulated reawrd
        self.action_t = tf.placeholder(tf.float32, [None, ACTION_DIM], name='action_t')

    def create_network(self):
        fc_p_1 = tf.layers.dense(self.state_t, 10, activation=tf.nn.relu, name='fc_p_1')
        fc_p_2 = tf.layers.dense(fc_p_1, 10, activation=tf.nn.relu, name='fc_p_2')
        self.logits_p = tf.layers.dense(fc_p_2, ACTION_DIM, activation=None, name='logits_p')
        self.softmax_p = tf.nn.softmax(self.logits_p)

        fc_v_1 = tf.layers.dense(self.state_t, 10, activation=tf.nn.relu, name='fc_v_1')
        fc_v_2 = tf.layers.dense(fc_v_1, 10, activation=tf.nn.relu, name='fc_v_2')
        self.logits_v = tf.layers.dense(fc_v_2, 1, activation=None, name='logits_v')

    def create_train_op(self):
        self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
        self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_t, axis=1)
        self.cost_p = -tf.reduce_sum(self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v)), axis=0)
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.cost_all = self.cost_p + self.cost_v 

        self.opt = tf.train.AdamOptimizer(0.01)
        self.train_op = self.opt.minimize(self.cost_all)

    def predict_p_and_v_single(self, state_t):
        prediction_p = self.sess.run(self.softmax_p, feed_dict={self.state_t: state_t[np.newaxis, :]})
        prediction_v = self.sess.run(self.logits_v, feed_dict={self.state_t: state_t[np.newaxis, :]})
        return prediction_p[0], prediction_v[0]

    def train(self, state_t, y_r, action_t):
        self.sess.run(self.train_op, feed_dict={self.state_t: state_t, self.y_r: y_r, self.action_t: action_t})