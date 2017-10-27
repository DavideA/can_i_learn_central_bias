import tensorflow as tf


def add_activations_to_summary(activations):
    for i, V in enumerate(activations):
        batchsize, iy, ix, channels = [int(x) for x in V.get_shape()]

        # The following trick to compute rows and columns is very neat
        # but works only if the number of channels is 2^n. This is not our case.
        #    def equal_divisors(n):
        #        d1 = 2
        #        d2 = n // d1
        #        while d2 > d1:
        #            d1 *= 2
        #            d2 = n // d1
        #        return d1, d2
        #    cx, cy = equal_divisors(channels)

        # Just compute a single column, all activations stacked one under the other
        cx, cy = 1, channels

        V = tf.reshape(V, (batchsize, iy, ix, cy, cx))

        V = tf.transpose(V, (0, 3, 1, 4, 2))
        V = tf.reshape(V, (batchsize, cy * iy, cx * ix, 1))

        name = 'activations_{:02d}'.format(i)
        tf.summary.image(name, V)


def add_histograms_to_summary(activations):

    for i, V in enumerate(activations):

        name = 'histogram_{:02d}'.format(i)
        tf.summary.histogram(name, V)
