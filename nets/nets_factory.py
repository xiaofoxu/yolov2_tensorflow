# coding=utf-8
"""contains a factory for building various models"""

import tensorflow as tf
import functools
from nets import YOLO2

slim = tf.contrib.slim

networks_map = {
    'darknet19': YOLO2.darknet19
}

arg_scopes_map = {

}


def get_network_fn(name, num_classes, is_training):
    if name not in networks_map:
        raise ValueError('name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fu(images):
        return func(images, num_classes, is_training=is_training)
    return network_fu
