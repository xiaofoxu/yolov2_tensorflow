# coding=utf-8
from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)