"""
See: https://fomoro.com/tools/receptive-fields/#3,1,1,SAME;2,2,1,VALID;3,1,1,SAME;2,2,1,VALID;3,1,1,SAME;3,1,1,SAME;
3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME
"""
import tensorflow as tf


conv2d = tf.layers.conv2d
pool2d = tf.layers.max_pooling2d
relu   = tf.nn.relu
sigmoid = tf.nn.sigmoid


def model_inference(X):

    _, height, width, _ = X.get_shape()

    h = conv2d(X, 16, [3, 3], padding='same', activation=relu)
    h = pool2d(h, pool_size=[2, 2], strides=[2, 2], padding='same')

    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = pool2d(h, pool_size=[2, 2], strides=[2, 2], padding='same')

    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)
    h = conv2d(h, 16, [3, 3], padding='same', activation=relu)

    h = conv2d(h, 1, [1, 1], padding='same', activation=None)

    return h
