import numpy as np
import tensorflow as tf

INPUT_DIM = 4
ACTION_DIM = 2

DISCOUNT_FACTOR = 0.99

class Network(object):
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
        self.state_t = tf.placeholder(tf.float32, [None, INPUT_DIM], name='state_t')
        self.action_t = tf.placeholder(tf.int32, [None], name="action_t")
        self.reward_tp1 = tf.placeholder(tf.float32, [None], name="reward_tp1")
        self.state_tp1 = tf.placeholder(tf.float32, [None, INPUT_DIM], name="state_tp1") 
        self.done = tf.placeholder(tf.bool, [None], name="done")

    def create_network(self):
        with tf.variable_scope("q_network"):
            fc_1 = tf.layers.dense(self.state_t, 10, activation=tf.nn.relu, name='fc_1')
            fc_2 = tf.layers.dense(fc_1, 10, activation=tf.nn.relu, name='fc_2')
            self.q = tf.layers.dense(fc_2, ACTION_DIM, activation=None, name='q')
        self.q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')

        with tf.variable_scope("target_q_network"):
            fc_1 = tf.layers.dense(self.state_tp1, 10, activation=tf.nn.relu, name='fc_1')
            fc_2 = tf.layers.dense(fc_1, 10, activation=tf.nn.relu, name='fc_2')
            self.target_q = tf.layers.dense(fc_2, ACTION_DIM, activation=None, name='target_q')
        self.target_q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q_network')

        self.update_target_q_network_op = tf.group(
            *[tq.assign(q) for q, tq in zip(
                sorted(self.q_network_vars, key=lambda v: v.name),
                sorted(self.target_q_network_vars, key=lambda v: v.name))])

    def create_train_op(self):
        self.q_acted = tf.reduce_sum(self.q * tf.one_hot(self.action_t, ACTION_DIM), 1) # Q(S_{t}, A_{t})
        # if done -> true, target becomes zero
        self.target = self.reward_tp1 + DISCOUNT_FACTOR \
            * (1. - tf.cast(self.done, tf.float32)) * tf.reduce_max(self.target_q, 1)
        self.cost = tf.reduce_mean(tf.squared_difference(self.q_acted, tf.stop_gradient(self.target)))

        self.opt = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = self.opt.minimize(self.cost, var_list=self.q_network_vars)

    def update_target_q_network(self):
        self.sess.run(self.update_target_q_network_op)

    def predict_q_single(self, state_t):
        return self.sess.run([self.q], feed_dict={self.state_t: state_t[np.newaxis, :]})

    def train(self, state_t, action_t, reward_tp1, state_tp1, done):
        feed_dict = {
            self.state_t: state_t,
            self.action_t: action_t,
            self.reward_tp1: reward_tp1,
            self.state_tp1: state_tp1,
            self.done: done
        }
        self.sess.run(self.train_op, feed_dict=feed_dict)
