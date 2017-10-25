import os
import tensorflow as tf
import numpy as np
import time
from model import model_inference
from data_io import get_batch_uniform_gt_filled
from os.path import exists, join


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Parameters
    h, w, c = 128, 128, 3
    batchsize = 4
    data_mode = 'uniform'
    square_side = 4
    padding_mode = 'constant'
    use_bias = True

    # Logs
    logs_path = join('learn_bias_logs', '{}'.format(time.time()))
    if not exists(logs_path):
        os.makedirs(logs_path)

    # Placeholders
    X = tf.placeholder(shape=(batchsize, h, w, c), dtype=tf.float32)
    Y = tf.placeholder(shape=(batchsize, 32, 32, 1), dtype=tf.float32)
    Y /= tf.reduce_sum(Y)

    # Model
    Z = model_inference(X)

    loss = tf.reduce_mean(tf.square(Y - Z))

    optim_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Summaries
    loss_s = tf.summary.scalar('loss', loss)
    tf.summary.image('X', X)
    tf.summary.image('Y', Y)
    tf.summary.image('Z', Z)
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

            X_num, Y_num = get_batch_uniform_gt_filled(shape=(h, w, c),
                                                       batchsize=batchsize,
                                                       square_side=square_side)

            loss_num, _ = sess.run([loss, optim_step], feed_dict={X: X_num, Y: Y_num})

            if counter % 200 == 0:
                summary_writer.add_summary(sess.run(merged_summary_op, feed_dict={X: X_num, Y: Y_num}),
                                           global_step=counter)
            else:
                summary_writer.add_summary(sess.run(loss_summary_op, feed_dict={X: X_num, Y: Y_num}),
                                           global_step=counter)

            counter += 1

            print(loss_num)
