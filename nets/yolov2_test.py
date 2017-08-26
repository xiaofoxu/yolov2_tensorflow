# coding=utf-8
import numpy as np
import tensorflow as tf
import yolov2 as yl


if __name__ == '__main__':
    inputs = np.ones((1, 416, 416, 3), dtype=np.float32)
    yolo = yl.YOLO2()
    net = yolo.darknet19(inputs, 1)
    print(net)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(net)
    all_var = yl.slim.get_variables()
    for v in all_var:
        print(v.name, v.get_shape())
    print(yolo.featureLayer)
