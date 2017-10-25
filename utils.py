import tensorflow as tf


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
