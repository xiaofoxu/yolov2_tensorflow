# coding=utf-8

import tensorflow as tf
import numpy as np
import os.path

slim = tf.contrib.slim

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DEVICE = 'CPU:0'
ALPHA = 0.1
outlist = []


class YOLO2:
    def __init__(self):
        self.featureLayer = None

    def darknet19(self, inputs, batch_size):
        layer0 = self.conv2d(inputs, 32, [3, 3], 1, scope='conv0')
        layer1 = slim.max_pool2d(layer0, [2, 2], scope='pool1')

        layer2 = self.conv2d(layer1, 64, [3, 3], 1, scope='conv2')
        layer3 = slim.max_pool2d(layer2, [2, 2], scope='pool3')

        layer4 = self.conv2d(layer3, 128, [3, 3], 1, scope='conv4')
        layer5 = self.conv2d(layer4, 64, [1, 1], 1, scope='conv5')
        layer6 = self.conv2d(layer5, 128, [3, 3], 1, scope='conv6')
        layer7 = slim.max_pool2d(layer6, [2, 2], scope='pool7')

        layer8 = self.conv2d(layer7, 256, [3, 3], 1, scope='conv8')
        layer9 = self.conv2d(layer8, 128, [1, 1], 1, scope='conv9')
        layer10 = self.conv2d(layer9, 256, [3, 3], 1, scope='conv10')
        layer11 = slim.max_pool2d(layer10, [2, 2], scope='pool11')

        layer12 = self.conv2d(layer11, 512, [3, 3], 1, scope='conv12')
        layer13 = self.conv2d(layer12, 256, [1, 1], 1, scope='conv13')
        layer14 = self.conv2d(layer13, 512, [3, 3], 1, scope='conv14')
        layer15 = self.conv2d(layer14, 256, [1, 1], 1, scope='conv15')
        layer16 = self.conv2d(layer15, 512, [3, 3], 1, scope='conv16')
        layer17 = slim.max_pool2d(layer16, [2, 2], scope='pool17')

        layer18 = self.conv2d(layer17, 1024, [3, 3], 1, scope='conv18')
        layer19 = self.conv2d(layer18, 512, [1, 1], 1, scope='conv19')
        layer20 = self.conv2d(layer19, 1024, [3, 3], 1, scope='conv20')
        layer21 = self.conv2d(layer20, 512, [1, 1], 1, scope='conv21')
        layer22 = self.conv2d(layer21, 1024, [3, 3], 1, scope='conv22')

        layer23 = self.conv2d(layer22, 1000, [1, 1], 1, scope='conv23')
        self.featureLayer = layer23
        layer24 = slim.avg_pool2d(layer23, [layer23.get_shape()[1], layer23.get_shape()[2]], scope='conv24')
        return layer24

    def leaky_relu(self, inputs):
        return tf.maximum(inputs, inputs * ALPHA)

    def conv2d(self, inputs, filters, kernel_size, stride, scope):
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=None,
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'scale': True,
                    'center': False,  # 公式中的bias，偏移量
                    'epsilon': 0.0000001
                },
        ):
            part1 = slim.conv2d(inputs, filters, kernel_size, stride, padding='SAME', scope=scope)
            part3 = slim.bias_add(part1, scope=scope)
            part4 = self.leaky_relu(part3)
            return part4

    def reorg(self, inputs, stride):
        outputs_1 = inputs[:, ::stride, ::stride, :]
        outputs_2 = inputs[:, ::stride, 1::stride, :]
        outputs_3 = inputs[:, 1::stride, ::stride, :]
        outputs_4 = inputs[:, 1::stride, 1::stride, :]
        output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis=-1)
        return output

