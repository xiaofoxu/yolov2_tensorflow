# coding=utf-8
import tensorflow as tf
from nets import YOLO2
slim = tf.contrib.slim

'''test for yolo2'''


class DarknetTest(tf.test.TestCase):
    def testBuild(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = YOLO2.darknet19(inputs, num_classes)
            self.assertEquals(logits.op.name, 'darknet19/conv24/AvgPool')
            self.assertListEqual(logits.get_shape().as_list(), [batch_size, 1, 1, num_classes])

    def testEndPoints(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            _, endpoints = YOLO2.darknet19(inputs, num_classes)
            expected_names = ['darknet19/conv0',
                              'darknet19/pool1',
                              'darknet19/conv2',
                              'darknet19/pool3',
                              'darknet19/conv4',
                              'darknet19/conv5',
                              'darknet19/conv6',
                              'darknet19/pool7',
                              'darknet19/conv8',
                              'darknet19/conv9',
                              'darknet19/conv10',
                              'darknet19/pool11',
                              'darknet19/conv12',
                              'darknet19/conv13',
                              'darknet19/conv14',
                              'darknet19/conv15',
                              'darknet19/conv16',
                              'darknet19/pool17',
                              'darknet19/conv18',
                              'darknet19/conv19',
                              'darknet19/conv20',
                              'darknet19/conv21',
                              'darknet19/conv22',
                              'darknet19/conv23'
                              ]
            self.assertSetEqual(set(endpoints.keys()), set(expected_names))

    def testModelVariables(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            YOLO2.darknet19(inputs, num_classes)
            vari = slim.get_model_variables()
            pass

    def testEvaluation(self):
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            eval_inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = YOLO2.darknet19(eval_inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(), [batch_size, 1, 1, num_classes])
            predictions = tf.argmax(logits, -1)
            self.assertListEqual(predictions.get_shape().as_list(), [batch_size, 1, 1])

    def testTrainEvalWithReuse(self):
        train_batch_size = 2
        eval_batch_size = 1
        train_height, train_width = 224, 224
        eval_height, eval_width = 256, 256
        num_classes = 1000
        with self.test_session():
            train_inputs = tf.random_uniform(
                (train_batch_size, train_height, train_width, 3))
            logits, _ = YOLO2.darknet19(train_inputs)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [train_batch_size, 1, 1, num_classes])
            tf.get_variable_scope().reuse_variables()
            eval_inputs = tf.random_uniform(
                (eval_batch_size, eval_height, eval_width, 3))
            logits, _ = YOLO2.darknet19(eval_inputs)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [eval_batch_size, 1, 1, num_classes])
            logits = tf.reduce_mean(logits, [1, 2])
            predictions = tf.argmax(logits, 1)
            self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])

    def testForward(self):
        batch_size = 1
        height, width = 224, 224
        with self.test_session() as sess:
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = YOLO2.darknet19(inputs)
            sess.run(tf.global_variables_initializer())
            output = sess.run(logits)
            self.assertTrue(output.any())
            print(output)

if __name__ == '__main__':
    tf.test.main()
