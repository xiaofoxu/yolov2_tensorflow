# coding=utf-8
"""
provides utilities to preprocess images for the Inception networks
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
    """
    Computes func(x, sel), with sel sampled from [0...num_cases-1]
    :param x: input Tensor
    :param func:  Python function to apply.
    :param num_cases: Python int32, number of cases to sample sel from.
    :return:  The result of func(x, sel), where func receives the value of the selector as a python integer,
            but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_radio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0), max_attempts=100, scope=None):
    """
    Generates cropped_image using a one of the bboxes randomly distorted.
    :param image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    :param bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged  as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    :param min_object_covered:  An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    :param aspect_radio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    :param area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    :param max_attempts:  An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    :param scope:  Optional scope for name_scope.
    :return:  A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        '''
        Each bounding box has shape [1, num_boxes, box coords] and
        the coordinates are ordered [ymin, xmin, ymax, xmax].
          # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        '''
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox,
                                                                               min_object_covered=min_object_covered,
                                                                               aspect_ratio_range=aspect_radio_range,
                                                                               area_range=area_range,
                                                                               max_attempts=max_attempts,
                                                                               use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox, fast_mode=True, scope=None):
    '''
    Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label. Additionally it would create image_summaries to display the different
    transformations applied to the image.
    :param image:
    :param height:
    :param width:
    :param bbox:
    :param fast_mode:
    :param scope:
    :return:
    '''
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image_with_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        tf.summary.image('image_with_bounding_boxes', image_with_bbox)
        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distorted_bbox)
        tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)
        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases)

        tf.summary.image('cropped_resized_image',
                         tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors. There are 4 ways to do it.
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases=4)

        tf.summary.image('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


def preprocess_for_eval(image, height, width, central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True):
    if is_training:
        return preprocess_for_train(image, height, width, bbox, fast_mode)
    else:
        return preprocess_for_eval(image, height, width)