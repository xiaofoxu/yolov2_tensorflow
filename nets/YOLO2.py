# coding=utf-8
import tensorflow as tf
slim = tf.contrib.slim

ALPHA = 0.1


def leaky_relu(inputs):
    return tf.maximum(inputs, inputs * ALPHA)


def conv2d(inputs, filters, kernel_size, stride, scope):
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
        part4 = leaky_relu(part3)
        return part4


def reorg(inputs, stride):
    outputs_1 = inputs[:, ::stride, ::stride, :]
    outputs_2 = inputs[:, ::stride, 1::stride, :]
    outputs_3 = inputs[:, 1::stride, ::stride, :]
    outputs_4 = inputs[:, 1::stride, 1::stride, :]
    output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis=-1)
    return output


def darknet19(inputs, num_classes=1000, is_training=True, scope='darknet19'):
    end_points = {}
    with tf.variable_scope(scope, 'darknet19') as sc:
        layer0 = conv2d(inputs, 32, [3, 3], 1, scope='conv0')
        end_points[sc.name + '/conv0'] = layer0
        layer1 = slim.max_pool2d(layer0, [2, 2], scope='pool1')
        end_points[sc.name + '/pool1'] = layer1

        layer2 = conv2d(layer1, 64, [3, 3], 1, scope='conv2')
        end_points[sc.name + '/conv2'] = layer2
        layer3 = slim.max_pool2d(layer2, [2, 2], scope='pool3')
        end_points[sc.name + '/pool3'] = layer3

        layer4 = conv2d(layer3, 128, [3, 3], 1, scope='conv4')
        end_points[sc.name + '/conv4'] = layer4
        layer5 = conv2d(layer4, 64, [1, 1], 1, scope='conv5')
        end_points[sc.name + '/conv5'] = layer5
        layer6 = conv2d(layer5, 128, [3, 3], 1, scope='conv6')
        end_points[sc.name + '/conv6'] = layer6
        layer7 = slim.max_pool2d(layer6, [2, 2], scope='pool7')
        end_points[sc.name + '/pool7'] = layer7

        layer8 = conv2d(layer7, 256, [3, 3], 1, scope='conv8')
        end_points[sc.name + '/conv8'] = layer8
        layer9 = conv2d(layer8, 128, [1, 1], 1, scope='conv9')
        end_points[sc.name + '/conv9'] = layer9
        layer10 = conv2d(layer9, 256, [3, 3], 1, scope='conv10')
        end_points[sc.name + '/conv10'] = layer10
        layer11 = slim.max_pool2d(layer10, [2, 2], scope='pool11')
        end_points[sc.name + '/pool11'] = layer11

        layer12 = conv2d(layer11, 512, [3, 3], 1, scope='conv12')
        end_points[sc.name + '/conv12'] = layer12
        layer13 = conv2d(layer12, 256, [1, 1], 1, scope='conv13')
        end_points[sc.name + '/conv13'] = layer13
        layer14 = conv2d(layer13, 512, [3, 3], 1, scope='conv14')
        end_points[sc.name + '/conv14'] = layer14
        layer15 = conv2d(layer14, 256, [1, 1], 1, scope='conv15')
        end_points[sc.name + '/conv15'] = layer15
        layer16 = conv2d(layer15, 512, [3, 3], 1, scope='conv16')
        end_points[sc.name + '/conv16'] = layer16
        layer17 = slim.max_pool2d(layer16, [2, 2], scope='pool17')
        end_points[sc.name + '/pool17'] = layer17

        layer18 = conv2d(layer17, 1024, [3, 3], 1, scope='conv18')
        end_points[sc.name + '/conv18'] = layer18
        layer19 = conv2d(layer18, 512, [1, 1], 1, scope='conv19')
        end_points[sc.name + '/conv19'] = layer19
        layer20 = conv2d(layer19, 1024, [3, 3], 1, scope='conv20')
        end_points[sc.name + '/conv20'] = layer20
        layer21 = conv2d(layer20, 512, [1, 1], 1, scope='conv21')
        end_points[sc.name + '/conv21'] = layer21
        layer22 = conv2d(layer21, 1024, [3, 3], 1, scope='conv22')
        end_points[sc.name + '/conv22'] = layer22

        layer23 = conv2d(layer22, num_classes, [1, 1], 1, scope='conv23')
        end_points[sc.name + '/conv23'] = layer23

        layer24 = slim.avg_pool2d(layer23, [layer23.get_shape()[1], layer23.get_shape()[2]], scope='conv24')
        return layer24, end_points
