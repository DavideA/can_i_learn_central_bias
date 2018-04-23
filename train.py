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
    dataset = 'noise'  # [`uniform`, `uniform_with_bias`, `noise`, `noise_with_bias`, `dreyeve`]
    grayscale = False               # warning! This only works for dreyeve
    dest_receptive_field = 114
    with_crops = True  # a crop is always half the size of the input

    # Set number of channels
    c = 1 if grayscale and dataset == 'dreyeve' else 3

    # Logs
    dataset_str = dataset if dataset != 'dreyeve' else '{}_{}'.format(dataset, 'grayscale' if grayscale else 'rgb')
    logs_path = join('/', 'majinbu', 'public', 'learn_bias_logs',
                     '{}_{}'.format(dataset_str, dest_receptive_field))

    # if cropping,
    if with_crops:
        # adjust sizes
        h_train, w_train, gt_h_train, gt_w_train = [x // 2 for x in (h, w, gt_h, gt_w)]
        # adjust logdir
        logs_path += '_crops'
    else:
        h_train, w_train, gt_h_train, gt_w_train = [h, w, gt_h, gt_w]

    if not exists(logs_path):
        os.makedirs(logs_path)

    # Placeholders
    X_train = tf.placeholder(shape=(batchsize, h_train, w_train, c), dtype=tf.float32)
    Y_train = tf.placeholder(shape=(batchsize, gt_h_train, gt_w_train, 1), dtype=tf.float32)
    Y_train /= tf.reduce_sum(Y_train)
    X_test = tf.placeholder(shape=(batchsize, h, w, c), dtype=tf.float32)
    Y_test = tf.placeholder(shape=(batchsize, gt_h, gt_w, 1), dtype=tf.float32)
    Y_test /= tf.reduce_sum(Y_test)

    # Model
    Z_train, activations = model_inference(X_train, dest_receptive_field=dest_receptive_field, total_filters_budget=256)
    Z_test, _ = model_inference(X_test, dest_receptive_field=dest_receptive_field, total_filters_budget=256, reuse=True)

    # Loss
    loss = tf.reduce_mean(tf.square(Y_train - Z_train))

    # Optimizer
    optim_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Summaries
    loss_s = tf.summary.scalar('loss', loss)
    tf.summary.image('X', X_test)
    tf.summary.image('Y', Y_test)
    tf.summary.image('Z', tf.clip_by_value(Z_test, 0, 1000))
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

        for _ in range(0, 2500):

            X_num_tr, Y_num_tr = get_batch(which_dataset=dataset, batchsize=batchsize,
                                           square_side=square_side, grayscale=grayscale,
                                           with_crops=with_crops)

            loss_num, _ = sess.run([loss, optim_step], feed_dict={X_train: X_num_tr, Y_train: Y_num_tr})

            summary_writer.add_summary(sess.run(loss_summary_op, feed_dict={X_train: X_num_tr, Y_train: Y_num_tr}),
                                       global_step=counter)

            if counter % 200 == 0:
                X_num_te, Y_num_te = get_batch(which_dataset=dataset, batchsize=batchsize,
                                         square_side=square_side, grayscale=grayscale,
                                         with_crops=False)

                summary_writer.add_summary(sess.run(merged_summary_op, feed_dict={
                    X_train: X_num_tr, Y_train: Y_num_tr,
                    X_test: X_num_te, Y_test: Y_num_te
                }),
                                           global_step=counter)

            counter += 1
