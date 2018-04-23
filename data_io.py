import numpy as np

from os.path import join

import skimage.io as io
from skimage.transform import resize
from skimage.color import rgb2gray

import warnings

h, w = 128, 128
gt_h, gt_w = 32, 32


def create_square_filled(square_side):
    gt = np.zeros(shape=(gt_h, gt_w, 1), dtype=np.float32)

    assert square_side > 0

    center_h, center_w = gt_h // 2, gt_w // 2

    square_start_h = center_h - square_side // 2
    square_end_h   = center_h + square_side // 2
    square_start_w = center_w - square_side // 2
    square_end_w   = center_w + square_side // 2
    gt[square_start_h: square_end_h, square_start_w: square_end_w, :] = 1.

    return gt


def get_batch_uniform(batchsize, square_side):

    X = np.stack([np.ones(shape=(h, w, 3)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    Y = np.stack([create_square_filled(square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_uniform_x_bias(batchsize, square_side):

    # X = np.random.rand(batchsize, h, w, c)
    X = np.stack([np.ones(shape=(h, w, 3)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    # Add a bias in the input
    # Bias kind: two lines
    # for x in X:
    #     x[:, 32] = 0.25
    #     x[32, :] = 0.1
    # Bias kind: little square in top left corner
    for x in X:
        x[20:30, 20:30] = 0.25

    Y = np.stack([create_square_filled(square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_noise(batchsize, square_side):

    X = np.random.rand(batchsize, h, w, 3)

    Y = np.stack([create_square_filled(square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_noise_x_bias(batchsize, square_side):

    X = np.random.rand(batchsize, h, w, 3)

    # Add a bias in the input
    # Bias kind: two lines
    # for x in X:
    #     x[:, 32] = 0.25
    #     x[32, :] = 0.1
    # Bias kind: little square in top left corner
    for x in X:
        x[20:30, 20:30] = 0.25

    Y = np.stack([create_square_filled(square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_dreyeve(batchsize, square_side, grayscale=False):

    dreyeve_root = '/majinbu/public/DREYEVE/DATA'

    X = [io.imread(join(dreyeve_root,
                        '{:02d}'.format(np.random.choice(range(1, 74))),
                        'frames', '{:06d}.jpg'.format(np.random.choice(range(0, 7500))))) for b in range(0, batchsize)]
    X = [resize(x, output_shape=(h, w)) for x in X]

    if grayscale:
        X = [rgb2gray(x) for x in X]
        X = [np.expand_dims(x, axis=-1) for x in X]

    X = np.stack(X, axis=0)

    Y = np.stack([create_square_filled(square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def crop_randomly(batch):

    X, Y = batch
    hc, wc, gt_hc, gt_wc = [i // 2 for i in (h, w, gt_h, gt_w)]

    newX, newY = [], []
    ratio = hc // gt_hc
    for x, y in zip(X, Y):
        # compute crop in gt which is smaller
        hy, wy = np.random.randint(0, gt_hc), np.random.randint(0, gt_wc)
        newY.append(y[hy:hy+gt_hc, wy:wy+gt_wc])

        hx, wx = [i*ratio for i in (hy, wy)]
        newX.append(x[hx:hx + hc, wx:wx + wc])

    return newX, newY


def get_batch(which_dataset, batchsize, square_side, grayscale, with_crops):

    assert which_dataset in ['uniform', 'uniform_with_bias', 'noise', 'noise_with_bias', 'dreyeve']

    if grayscale and which_dataset != 'dreyeve':
        warnings.warn('grayscale=True has no effect with which_dataset=={}'.format(which_dataset))

    if which_dataset == 'uniform':
        batch = get_batch_uniform(batchsize, square_side)
    elif which_dataset == 'uniform_with_bias':
        batch = get_batch_uniform_x_bias(batchsize, square_side)
    elif which_dataset == 'noise':
        batch = get_batch_noise(batchsize, square_side)
    elif which_dataset == 'noise_with_bias':
        batch = get_batch_noise_x_bias(batchsize, square_side)
    elif which_dataset == 'dreyeve':
        batch = get_batch_dreyeve(batchsize, square_side, grayscale)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(which_dataset))

    if with_crops:
        batch = crop_randomly(batch)

    return batch


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    X_num, Y_num = get_batch_uniform(batchsize=16, square_side=8)

    for y in Y_num:
        plt.imshow(np.squeeze(y))
        plt.show()
