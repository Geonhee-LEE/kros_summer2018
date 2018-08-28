import os
import cv2
import argparse
import datetime
import numpy as np
import tensorflow as tf
import yolo.config as cfg
import matplotlib.pyplot as plt
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
            # decayed_learning_rate = learning_rate * decay_rate ^ int(global_step / decay_step)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)
        
    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 1) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {:.6f}, '
                            'Loss: {:.3f}\nSpeed: {:.3f}s/iter, '
                            'Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def display_gt(detect_gt):
    img_size = cfg.IMAGE_SIZE
    class_name = cfg.CLASSES
    img = cv2.imread(detect_gt['imname'])
    img_h, img_w, _ = img.shape
    label_gt = detect_gt['label']      # 7 x 7 x 25, (25 = 1 (objective) + 4 (cx,cy, w,h)
    h_ratio = img_size / img_h
    w_ratio = img_size / img_w
    objectiveness = label_gt[:,:,0]  # 7x7--> Objectiveness
    boxes = label_gt[:, :, 1:5]      # 7 x 7 x 4, (cx,cy, w, h)
    classes = label_gt[:, :, 5:25]  # 7 x 7 x 20, (class)

    if detect_gt['flipped']:
        img = cv2.flip(img, 1)

    object_cells = np.nonzero(np.array(objectiveness == 1, dtype='bool'))
    object_boxes = boxes[object_cells[0],object_cells[1], :]
    object_classes = np.argmax(classes[object_cells[0],object_cells[1],:],axis=1)

    for i in range(len(object_boxes)):
        x1 = int((int(object_boxes[i][0]) - int(object_boxes[i][2] / 2)) / w_ratio)
        y1 = int((int(object_boxes[i][1]) - int(object_boxes[i][3] / 2)) / h_ratio)
        x2 = int((int(object_boxes[i][0]) + int(object_boxes[i][2] / 2)) / w_ratio)
        y2 = int((int(object_boxes[i][1]) + int(object_boxes[i][3] / 2)) / h_ratio)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (125, 125, 125), -1)
        lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
        cv2.putText(img, class_name[object_classes[i]],(x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType)

    cv2.imshow('Image', img)
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    yolo = YOLONet(True)
    data = pascal_voc('train')

    for i in range(10):
        display_gt(data.gt_labels[i])

    solver = Solver(yolo, data)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':
    main()
