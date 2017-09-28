import tensorflow as tf


def cnn_model_valid(X, encoder_filters, decoder_filters, use_bias, padding_mode):

    activations = []

    _, height, width, _ = X.get_shape()

    network_depth = len(encoder_filters)
    num_pooling = 4

    h = X

    # encoding
    for i, filter_size in enumerate(encoder_filters):

        if i < num_pooling:
            last_strides = (2, 2)
        else:
            last_strides = (1, 1)

        # last_strides = (1, 1) if i >= len(encoder_filters) - num_pooling else (2, 2)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu,
                             strides=last_strides)

        activations.append(h)

    for i, filter_size in enumerate(decoder_filters):

        _, cur_h, cur_w, _ = h.get_shape()
        if i >= network_depth - num_pooling - 1:
            new_height = int(cur_h * 2)
            new_width = int(cur_w * 2)
            h = tf.image.resize_images(h, size=[int(new_height) + 6, int(new_width) + 6], method=1)
        else:
            h = tf.image.resize_images(h, size=[int(cur_h) + 6, int(cur_w) + 6], method=1)

        # new_height = int(height) // 2 ** (network_depth - i - 3)  # last encoding block has strides=(1, 1)
        # new_width = int(width) // 2 ** (network_depth - i - 3)  # last encoding block has strides=(1, 1)
        # h = tf.image.resize_images(h, size=[new_height + 6, new_width + 6], method=1)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        activations.append(h)

    h = tf.image.resize_images(h, size=[int(height), int(width)], method=1)

    Z = tf.layers.conv2d(h, 1, kernel_size=(1, 1), use_bias=use_bias, activation=tf.nn.relu)

    return Z, activations


def cnn_model(X, encoder_filters, decoder_filters, use_bias, padding_mode):

    activations = []

    h = X

    # encoding
    for i, filter_size in enumerate(encoder_filters):

        last_strides = (1, 1) if i + 1 == len(encoder_filters) else (2, 2)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu,
                             strides=last_strides)

        activations.append(h)

    for i, filter_size in enumerate(decoder_filters):

        _, height, width, c = h.get_shape()
        h = tf.image.resize_images(h, size=[int(height) * 2, int(width) * 2], method=1)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
        h = tf.layers.conv2d(h, filter_size, kernel_size=(3, 3), use_bias=use_bias, activation=tf.nn.relu)

        activations.append(h)

    Z = tf.layers.conv2d(h, 1, kernel_size=(1, 1), use_bias=use_bias, activation=tf.nn.relu)

    return Z, activations
