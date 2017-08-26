# coding=utf-8
import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'cifar10_%s.tfrecord'
SPLITS_TO_SIZE = {'train': 50000, 'test': 10000}
_NUM_CLASSES = 10
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9.',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in SPLITS_TO_SIZE:
        raise ValueError('split name %s was not recognized.' % split_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZE[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labes_to_names=labels_to_names
    )
