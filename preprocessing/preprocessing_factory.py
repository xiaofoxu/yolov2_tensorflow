# coding=uf-8
import tensorflow as tf
from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import  lenet_preprocessing
from preprocessing import  vgg_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
    preprocessing_fn_map = {
        'cifarnet': cifarnet_preprocessing,
        'inception': inception_preprocessing,
        'inception_v1': inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'lenet': lenet_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'resnet_v1_200': vgg_preprocessing,
        'resnet_v2_50': vgg_preprocessing,
        'resnet_v2_101': vgg_preprocessing,
        'resnet_v2_152': vgg_preprocessing,
        'resnet_v2_200': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(image, output_height, output_width, is_training=is_training,
                                                           **kwargs)
    return preprocessing_fn
