import numpy as np
import tensorflow as tf

import os
from os.path import exists, join

from model import model_inference

from data_io import get_batch, h, w, gt_h, gt_w

from utils import add_activations_to_summary
from utils import add_histograms_to_summary


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Parameters
    batchsize   = 8
    square_side = 4
    dataset = 'noise_with_bias'  # [`uniform`, `uniform_with_bias`, `noise`, `noise_with_bias`, `dreyeve`]
    grayscale = False        # warning! This only works for dreyeve
    dest_receptive_field = 114

    # Set number of channels
    c = 1 if grayscale and dataset == 'dreyeve' else 3

    # Logs
    dataset_str = dataset if dataset != 'dreyeve' else '{}_{}'.format(dataset, 'grayscale' if grayscale else 'rgb')
    logs_path = join('/', 'majinbu', 'public', 'learn_bias_logs',
                     '{}_{}'.format(dataset_str, dest_receptive_field))
    if not exists(logs_path):
        os.makedirs(logs_path)

    # Placeholders
    X = tf.placeholder(shape=(batchsize, h, w, c), dtype=tf.float32)
    Y = tf.placeholder(shape=(batchsize, gt_h, gt_w, 1), dtype=tf.float32)
    Y /= tf.reduce_sum(Y)

    # Model
    Z, activations = model_inference(X, dest_receptive_field=dest_receptive_field, total_filters_budget=256)

    # Loss
    loss = tf.reduce_mean(tf.square(Y - Z))

    # Optimizer
    optim_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Summaries
    loss_s = tf.summary.scalar('loss', loss)
    tf.summary.image('X', X)
    tf.summary.image('Y', Y)
    tf.summary.image('Z', Z)
    add_activations_to_summary(activations)
    add_histograms_to_summary(activations)
    loss_summary_op = tf.summary.merge([loss_s])
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Print number of trainable parameters
    trainable_vars = np.sum([int(np.prod(v.get_shape())) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
    print('Trainable variables: {:,}'.format(trainable_vars))

    # Train
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        counter = 0

        while True:

            X_num, Y_num = get_batch(which_dataset=dataset, batchsize=batchsize,
                                     square_side=square_side, grayscale=grayscale)

            loss_num, _ = sess.run([loss, optim_step], feed_dict={X: X_num, Y: Y_num})

            if counter % 200 == 0:
                summary_writer.add_summary(sess.run(merged_summary_op, feed_dict={X: X_num, Y: Y_num}),
                                           global_step=counter)
            else:
                summary_writer.add_summary(sess.run(loss_summary_op, feed_dict={X: X_num, Y: Y_num}),
                                           global_step=counter)

            counter += 1

            print(loss_num)
