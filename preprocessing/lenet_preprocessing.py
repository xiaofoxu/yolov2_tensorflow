# coding=utf-8
import tensorflow as tf

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width, is_training):
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    return image
