# coding=utf-8
"""
provides utilities to preprocess images in CIFAR-10
"""
import tensorflow as tf

_PADDING = 4

slim = tf.contrib.slim


def preprocess_for_train(image, output_height, output_width, padding=_PADDING):
    """
    Preprocesses the given image for training. Note that the actual resizing scale is sampled from
    [resize_size_min, resize_size_max]
    :param image: A 'Tensor' representing an image of arbitrary size
    :param output_height: the height of the image after preprocessing
    :param output_width: the width of the image after preprocessing
    :param padding: The amound of padding before and after each dimension of the image.
    :return:  A preprocessed image.
    """
    tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    if padding > 0:
        image = tf.pad(image, [[padding, padding], [padding, padding]], 'CONSTANT')

    # randomly crop a [height, width] section of the image
    distorted_image = tf.random_crop(image, [output_height, output_width, 3])

    # randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels
    return tf.image.per_image_standardization(distorted_image)


def preprocess_for_eval(image, output_height, output_width):
    tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    resized_image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)
    tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))
    return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False):
    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)
