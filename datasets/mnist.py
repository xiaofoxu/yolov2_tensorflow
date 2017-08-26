# coding=utf-8
import os
import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim
_FILE_PATTERN = 'mnist_%s.tfrecord'
_SPLITS_TO_SIZES = {'train': 60000, 'test': 10000}
_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'a [28 x 28 x 1] grayscale image.',
    'label': 'a single integer between 0 and 9',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """
    Gets a dataset tuple with instructions for reading MNIST
    :param split_name: A train/test split name
    :param dataset_dir: the base directory of the dataset sources
    :param file_pattern: the file pattern to use when matching the dataset sources. It is assumed that the pattern contains
    a '%s' string so that the split name can be inserted
    :param reader: the tensorflow reader type
    :return: A 'Dataset' namedtuple
    :raise: ValueError: if 'split_name' is not a valid train/test split
    """
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            shape=[28, 28],
            channel=1
        ),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        num_classes=_NUM_CLASSES,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        labels_to_names=labels_to_names
    )
