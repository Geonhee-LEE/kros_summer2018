import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)          # C = 20
        self.image_size = cfg.IMAGE_SIZE            # 448
        self.cell_size = cfg.CELL_SIZE              # S = 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL    # B = 2
        self.threshold = cfg.THRESHOLD              # 0.2
        self.iou_threshold = cfg.IOU_THRESHOLD      # 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class   
            # S x S x C = 980  ---> 0:boundary1
            #    = Conditional Class Probablity, Pr(Class_i|Object)
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell
            # S x S x C + S x S x B = 980 + 98 = 1078
            # ---> boundary1:boundary2 = Confidence Score, Pr(Object)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # restore weights file
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def detect(self, img):
        img_h, img_w, _ = img.shape     # img_h, img_w, _
        inputs = cv2.resize(img, (self.image_size, self.image_size))
            # (448, 448, 3)
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)   # BGR --> RGB
        inputs = (inputs / 255.0) * 2.0 - 1.0       # normalization
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))  # Training 구조
            # (1, 448, 448, 3)
        result = self.detect_from_cvmat(inputs)   # Call YOLO net
            #[('class', cx, cy, w, h, prob), ('class', cx, cy, w, h, prob), ....]

        for i in range(len(result)):                 # Number of objects
            result[i][1] *= (img_w / self.image_size)    # Return to the original size
            result[i][2] *= (img_h / self.image_size)    # Return to the original size
            result[i][3] *= (img_w / self.image_size)    # Return to the original size
            result[i][4] *= (img_h / self.image_size)    # Return to the original size  +
        return result


    def detect_from_cvmat(self, input):     # Call YOLO net
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: input})
            # 1, 1470 = S x S x (B * 5 + C)

        return self.interpret_output(net_output[0])

    def interpret_output(self, output):

        # (S, S, B, C)
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))

        # Conditional Class Probablity, Pr(Class_i|Object)    
        class_probs = np.reshape(output[0:self.boundary1],(self.cell_size, self.cell_size, self.num_class))
            # (S, S, C)
        
        # Confidence Score, Pr(Object)
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
            # (S, S, B)
        
        # Bounding Box, (cx, xy, w, h) in the range [0, 1]
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
            # (S, S, B, 4)
        
        # interpret network output (cx, cy, w, h) using offset and compute the bounding box
        offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        # (S, S, B)
        # offset[:, :, 0]=  offset[:,:,1]
        #                 = array([[0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6],
        #                         [0, 1, 2, 3, 4, 5, 6]])

        offset_tran = np.transpose(offset, (1, 0, 2))
        # (S, S, B)
        # offset_tran[:, :, =]=offset_tran[:, :, 1]
        #                     = array([[0, 0, 0, 0, 0, 0, 0],
        #                             [1, 1, 1, 1, 1, 1, 1],
        #                             [2, 2, 2, 2, 2, 2, 2],
        #                             [3, 3, 3, 3, 3, 3, 3],
        #                             [4, 4, 4, 4, 4, 4, 4],
        #                             [5, 5, 5, 5, 5, 5, 5],
        #                             [6, 6, 6, 6, 6, 6, 6]])


        boxes[:, :, :, 0] += offset                 # cx
        boxes[:, :, :, 1] += offset_tran            # cy
        boxes[:, :, :, :2] /= self.cell_size        # cx and cy into [0,1]
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:]) # root(W) and root(H) into W and H
        boxes *= self.image_size                    #resize to the original image size

        # Class Specific Confidence Score
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        # filtering via class specific confidence score
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        # returns the cell, box and class id for valid object

        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2],:]
        # returns the bboxes (cx,cy,w,h) for valid object

        probs_filtered = probs[filter_mat_probs]
        # returns the class conditional prob. for valid object

        classes_num_filtered = np.argmax(
            probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        # returns the class num for valid object
        
        # non-maximal suppression
        # step-1: sorting class specific confidence score in a descending order
        argsort = np.argsort(probs_filtered)[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # step-2: filtering via iou
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])
            # (class, x, y, w, h, score)

        return result

    def iou(self, box1, box2):
        lr = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        tb = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if lr < 0 or tb < 0 else lr * tb
        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
        return inter / union

    def display_result(self, img, result):
        for i in range(len(result)):
            x1 = int(result[i][1]) - int(result[i][3] / 2)
            y1 = int(result[i][2]) - int(result[i][4] / 2)
            x2 = int(result[i][1]) + int(result[i][3] / 2)
            y2 = int(result[i][2]) + int(result[i][4] / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1 - 20),
                          (x2, y1), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.3f' % result[i][5],
                (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)
            
    def image_detector(self, imname, wait=0):
        image = cv2.imread(imname)
        result = self.detect(image)
        self.display_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    weight_file = "./data/weights/YOLO_small.ckpt"
    
    yolo = YOLONet(False)
    detector = Detector(yolo, weight_file)

    # detect from image file
    imname = './images/cat.jpg'
    #imname = './images/person.jpg'
    #imname = './images/000020.jpg'
    #imname = './images/000021.jpg'
    #imname = './images/000023.jpg'
    
    detector.image_detector(imname)
    

if __name__ == '__main__':
    main()
