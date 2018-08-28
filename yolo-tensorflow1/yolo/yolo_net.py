import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim



class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)                      # C = 20
        self.image_size = cfg.IMAGE_SIZE                        # 448  
        self.cell_size = cfg.CELL_SIZE                          # S = 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL                # B = 2   
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.boxes_per_cell * 5 + self.num_class)          # 1470 = S x S x (B * 5 + C)
        self.scale = self.image_size / self.cell_size           # 448/7 = 64
        self.boundary1 = self.cell_size * self.cell_size * self.num_class    # Class probability
            # S x S x C = 980   
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell           # Objectiveness
            # S x S x C + S x S x B = 980 + 98 = 1078

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
        
    def build_network(self,
                      images,
                      num_outputs,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha=0.1),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                # images (?, 448, 448, 3)
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                    # (?, 454, 454, 3)
                    # 454 = 3 + 448 + 3
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                    # (?, 224, 224, 64) = 454-7
                    # here, 'VALID' means no padding.
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                    # (?, 112, 112, 64)
                    # here, 'SAME' means zero-paddings.
                net = slim.conv2d(net, 192, 3, padding='SAME', scope='conv_4')
                    # (?, 112, 112, 192)
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                    # (?, 56, 56, 192)
                net = slim.conv2d(net, 128, 1, padding='SAME',scope='conv_6')
                    # (?, 56, 56, 192)
                net = slim.conv2d(net, 256, 3, padding='SAME',scope='conv_7')
                    # (?, 56, 56, 256)
                net = slim.conv2d(net, 256, 1, padding='SAME',scope='conv_8')
                    # (?, 56, 56, 256)
                net = slim.conv2d(net, 512, 3, padding='SAME',scope='conv_9')
                    # (?, 56, 56, 512)
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                    # (?, 28, 28, 256)
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                    # (?, 28, 28, 256)
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                    # (?, 28, 28, 256)
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                    # (?, 28, 28, 256)
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                    # (?, 28, 28, 512)
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                    # (?, 28, 28, 1024)
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                    # (?, 14, 14, 1024)
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                    # (?, 14, 14, 512)
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                    # (?, 14, 14, 1024)
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                    # (?, 14, 14, 512)
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                    # (?, 14, 14, 1024)
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                    # (?, 14, 14, 1024)
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                    # (?, 16, 16, 1024)
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                    # (?, 7, 7, 1024)
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                    # (?, 7, 7, 1024)
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                    # (?, 7, 7, 1024)
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                    # (?, 1024, 7, 7)
                net = slim.flatten(net, scope='flat_32')
                    # (?, 50176)
                net = slim.fully_connected(net, 512, scope='fc_33')
                    # (?, 512)
                net = slim.fully_connected(net, 4096, scope='fc_34')
                    # (?, 4096)
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                    # (?, 4096)
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
                    # (?, 1470)
        return net

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
                # 0:boundary1-1 = Conditional Class Probablity, Pr(Class_i|Object)
                # (?, 980) => (?, 7, 7, 20)

            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
                # boundary1:boundary2-1 = Confidence Score, Pr(Object)
                # (?, 98) => (? 7, 7, 2)

            predict_boxes_coord  = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
                # boundary2: = Bbox (cx,cy,w,h)
                # (?, 392) ==> (?, 7, 7, 2, 4)
                
            # response: contain objects or not in GT (labels)
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            # labels = <tf.Tensor 'Placeholder:0' shape=(?, 7, 7, 25) dtype=float32>
            # labels[..., 0] = labels[:,:,:,0]
            # increase one dimension

            # boxes: bbox location in GT (labels)
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # (?, 7, 7, 1, 4)

            # Bounding Box, normalized into the range [0, 1]
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            # (?, 7, 7, 2, 4)

            classes = labels[..., 5:]
                # (?, 7, 7, 20)

            offset = np.transpose(np.reshape(np.array(
                [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
                # (7, 7, 2)
                # (S, S, B)
                # offset[:, :, 0]=  offset[:,:,1]
                #                 = array([[0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6],
                #                         [0, 1, 2, 3, 4, 5, 6]])


            offset = tf.reshape(
                tf.constant(offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
                #  Constant --> Tensor in TF (1, 7, 7, 2)

            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
                # (?, 7, 7, 2)
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))


            predict_boxes = tf.stack(
                [(predict_boxes_coord[..., 0] + offset) / self.cell_size,
                 (predict_boxes_coord[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes_coord[..., 2]),
                 tf.square(predict_boxes_coord[..., 3])], axis=-1)

            # predict_boxes_coord[..., 0]=predict_boxes_coord[:,:,:,0]
            # <tf.Tensor 'loss_layer/strided_slice_10:0' shape=(45, 7, 7, 2) dtype=float32>

            # predict_boxes_coord[..., 1]=predict_boxes_coord[:,:,:,1]
            # <tf.Tensor 'loss_layer/strided_slice_11:0' shape=(45, 7, 7, 2) dtype=float32>



            iou_predict_truth = self.calc_iou(predict_boxes, boxes)
            # predict_boxes = <tf.Tensor 'loss_layer/stack:0' shape=(45, 7, 7, 2, 4) dtype=float32>
            # boxes (gt boxes) = <tf.Tensor 'loss_layer/truediv:0' shape=(45, 7, 7, 2, 4) dtype=float32>
            # iou_predict_truth = <tf.Tensor 'loss_layer/clip_by_value:0' shape=(45, 7, 7, 2) dtype=float32>


            # calculate I_obj tensor (?, S, S, B)
            object_mask = tf.reduce_max(iou_predict_truth, axis=3, keepdims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response


            # calculate I_noobj tensor (?, S, S, B)
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_coord = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)
                 
            # coord_loss
            coord_mask = tf.expand_dims(object_mask, -1)
            boxes_delta = coord_mask * (predict_boxes_coord - boxes_coord)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            # object_loss
            # object_delta = object_mask * (predict_scales - iou_predict_truth)
            # the original code according to the equation presented in the paper
            # response = <tf.Tensor 'loss_layer/Reshape_3:0' shape=(45, 7, 7, 1) dtype=float32>
            object_delta = object_mask * (predict_scales - response)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            # noobject_delta = noobject_mask * predict_scales
            # the original code
            noobject_delta = noobject_mask * (predict_scales - response)
                # according to the equation presented in the paper
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale


            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def calc_iou(self, boxes1, boxes2, scope='cals_iou'):
        '''Calculate IoUs
        Args:
          boxes1: 5-D tensor [_, S, S, B, 4] ===> (x_center, y_center, w, h)
          boxes2: 5-D tensor [_, S, S, B, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [_, S, S, B]
        '''
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            inter = tf.maximum(0.0, rd - lu)
            inter_square = inter[..., 0] * inter[..., 1]

            boxes1_square = boxes1[..., 2] * boxes1[..., 3]
            boxes2_square = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(boxes1_square + boxes2_square - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def leaky_relu(alpha=0.1):
    def op(x):
        return tf.nn.leaky_relu(x, alpha=alpha, name='leaky_relu')
    return op
