"""
See: https://fomoro.com/tools/receptive-fields/#3,1,1,SAME;2,2,1,VALID;3,1,1,SAME;2,2,1,VALID;3,1,1,SAME;3,1,1,SAME;
3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME;3,1,1,SAME
"""
import tensorflow as tf


conv2d  = tf.layers.conv2d
pool2d  = tf.layers.max_pooling2d
relu    = tf.nn.relu


def model_inference(X, dest_receptive_field, total_filters_budget, verbose=True, reuse=None):

    # List to store intermediate activations (to be shown in summaries)
    activations = []

    _, height, width, _ = X.get_shape()

    with tf.variable_scope('network', reuse=reuse):
        h = conv2d(X, 16, [3, 3], padding='same', activation=relu)       # receptive field now: 3
        h = pool2d(h, pool_size=[2, 2], strides=[2, 2], padding='same')  # receptive field now: 4
        activations.append(h)

        h = conv2d(h, 16, [3, 3], padding='same', activation=relu)       # receptive field now: 8
        h = pool2d(h, pool_size=[2, 2], strides=[2, 2], padding='same')  # receptive field now: 10
        activations.append(h)

        current_receptive_field = 10
        receptive_field_stride  = 8  # how much receptive field increases each conv from now on

        # Compute the number of convolutional layers to stack in order to reach
        # the desired receptive field on the input tensor
        num_conv_needed = (dest_receptive_field - current_receptive_field) // receptive_field_stride

        # In order to keep the number of parameters comparable for all models despite
        # the different number of layers, the number of filters learned by each layer
        # increases as the network gets shallower
        num_filters_each_conv = total_filters_budget // num_conv_needed

        # Stack conv layers as needed
        for _ in range(num_conv_needed):
            h = conv2d(h, num_filters_each_conv, [3, 3], padding='same', activation=relu)
            activations.append(h)
        Z = conv2d(h, 1, [1, 1], padding='same', activation=None)

    if verbose:
        print('Num of convolutional layers: {}'.format(num_conv_needed))
        print('Num of filters each conv: {}'.format(num_filters_each_conv))

    return Z, activations
