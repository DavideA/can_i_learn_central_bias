import tensorflow as tf
import numpy as np
import time
from model import model_inference
from data_io import get_batch, get_batch_uniform_gt_hole

from os.path import exists, join
import os


try:
    zrange = xrange
except NameError:
    zrange = range


def add_activations_to_summary(activations, batchsize):
    for i, V in enumerate(activations):
        iy, ix, channels = map(lambda x: int(x), V.get_shape()[1:])

        def equal_divisors(n):
            d1 = 2
            d2 = n // d1
            while d2 > d1:
                d1 *= 2
                d2 = n // d1
            return d1, d2

        cx, cy = equal_divisors(channels)
        V = tf.reshape(V, (batchsize, iy, ix, cy, cx))

        V = tf.transpose(V, (0, 3, 1, 4, 2))
        V = tf.reshape(V, (batchsize, cy * iy, cx * ix, 1))

        name = 'activations_{:02d}'.format(i)
        tf.summary.image(name, V)


def add_histograms_to_summary(activations):

    for i, V in enumerate(activations):

        name = 'histogram_{:02d}'.format(i)
        tf.summary.histogram(name, V)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    h, w, c = 128, 128, 3
    batchsize = 4
    data_mode = 'uniform'
    gaussian_var = 7500
    side_ext, side_int = 8, 4
    # encoder_filters = [128, 256, 512, 1024, 1024]
    # decoder_filters = [1024, 512, 256, 128]
    encoder_filters = [4, 8, 16, 32, 32]
    decoder_filters = [32, 16, 8, 8]
    padding_mode = 'constant'
    use_bias = True
    logs_path = join('learn_bias_logs',
                     'logs_big_ball',
                     'input_{}_padding_{}_{}'.format(data_mode.upper(), padding_mode.upper(), time.time()))

    translation = [0, 100]

    if not exists(logs_path):
        os.makedirs(logs_path)

    # model
    X = tf.placeholder(shape=(batchsize, h, w, c), dtype=tf.float32)
    Y = tf.placeholder(shape=(batchsize, 32, 32, 1), dtype=tf.float32)
    Y /= tf.reduce_sum(Y)

    Z = model_inference(X)

    # Z /= tf.reduce_sum(Z) + np.finfo(float).eps

    # loss = tf.reduce_mean(Y*tf.log(np.finfo(float).eps + Y/(Z + np.finfo(float).eps)))
    loss = tf.reduce_mean(tf.square(Y - Z))

    optim_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    loss_s = tf.summary.scalar('loss', loss)
    tf.summary.image('X', X)
    tf.summary.image('Y', Y)
    tf.summary.image('Z', Z)

    # add_activations_to_summary(activations, batchsize)
    # add_histograms_to_summary(activations)

    loss_summary_op = tf.summary.merge([loss_s])
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        trainable_vars = np.sum([int(np.prod(v.get_shape())) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        print('Trainable variables: {:,}'.format(trainable_vars))

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for counter in zrange(1, 10000000000):

            # X_num, Y_num = get_batch(shape=(h, w, c), batchsize=batchsize, gaussian_var=gaussian_var,
            #                          type=data_mode, translation=translation)
            X_num, Y_num = get_batch_uniform_gt_hole(shape=(h, w, c),
                                                     batchsize=batchsize,
                                                     side_ext=side_ext,
                                                     side_int=side_int)

            feed_dict = {X: X_num, Y: Y_num}
            loss_num, _ = sess.run([loss, optim_step], feed_dict=feed_dict)

            if counter % 200 == 0:
                summary_writer.add_summary(sess.run(merged_summary_op, feed_dict=feed_dict),
                                           global_step=counter)
            else:
                summary_writer.add_summary(sess.run(loss_summary_op, feed_dict=feed_dict),
                                           global_step=counter)

            print(loss_num)
