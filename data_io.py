import numpy as np
from scipy.stats import multivariate_normal

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

voc_root = '/majinbu/public/HIGHDIMFILTERING/datasets/VOC2012/VOC2012/JPEGImages'
voc_list = glob(join(voc_root, '*.jpg'))
voc_len = len(voc_list)

dreyeve_root = '/majinbu/public/DREYEVE/DATA'

cityscapes_root = '/majinbu/public/HIGHDIMFILTERING/datasets/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit'
cityscapes_list = glob(join(cityscapes_root, '**', '**', '*.png'))
cityscapes_len = len(cityscapes_list)


def create_square_with_hole(height, width, side_ext, side_int):
    gt = np.zeros(shape=(height, width, 1), dtype=np.float32)

    assert side_int < side_ext

    center_h, center_w = height // 2, width // 2

    side_ext = side_int  # SOLERA, fuck tf
    ext_square_start_h = center_h - side_ext // 2
    ext_square_end_h = center_h + side_ext // 2
    ext_square_start_w = center_w - side_ext // 2
    ext_square_end_w = center_w + side_ext // 2
    gt[ext_square_start_h: ext_square_end_h, ext_square_start_w: ext_square_end_w, :] = 1.

    #int_square_start_h = center_h - side_int // 2
    #int_square_end_h = center_h + side_int // 2
    #int_square_start_w = center_w - side_int // 2
    #int_square_end_w = center_w + side_int // 2
    #gt[int_square_start_h: int_square_end_h, int_square_start_w: int_square_end_w, :] = 0.

    return gt


def get_batch_cityscapes(shape, batchsize, gaussian_var, translation):

    h, w, c = shape

    idxs = np.random.choice(range(0, cityscapes_len), size=batchsize)
    X = np.stack([resize(io.imread(cityscapes_list[i]), output_shape=(h, w))for i in idxs], axis=0)

    pos = np.stack(np.mgrid[0:h:1, 0:w:1], axis=-1)
    h_t, w_t = translation
    pdf = multivariate_normal.pdf(pos, mean=[h // 2 + h_t, w // 2 + w_t], cov=[[gaussian_var, 0], [0, gaussian_var]])

    Y = np.stack([np.expand_dims(pdf, axis=-1) for b in range(0, batchsize)], axis=0)
    Y /= np.max(Y)

    return np.float32(X), np.float32(Y)


def get_batch_dreyeve(shape, batchsize, gaussian_var, translation):

    h, w, c = shape

    idxs = np.random.choice(range(0, voc_len), size=batchsize)
    X = np.stack([resize(io.imread(join(dreyeve_root,
                                        '{:02d}'.format(np.random.choice(range(1, 74))),
                                        'frames',
                                        '{:06d}.jpg'.format(np.random.choice(range(0, 7500)))))
                         , output_shape=(h, w)) for i in idxs], axis=0)

    pos = np.stack(np.mgrid[0:h:1, 0:w:1], axis=-1)
    h_t, w_t = translation
    pdf = multivariate_normal.pdf(pos, mean=[h // 2 + h_t, w // 2 + w_t], cov=[[gaussian_var, 0], [0, gaussian_var]])

    Y = np.stack([np.expand_dims(pdf, axis=-1) for b in range(0, batchsize)], axis=0)
    Y /= np.max(Y)

    return np.float32(X), np.float32(Y)


def get_batch_voc(shape, batchsize, gaussian_var, translation):

    h, w, c = shape

    idxs = np.random.choice(range(0, voc_len), size=batchsize)
    X = np.stack([resize(io.imread(voc_list[i]), output_shape=(h, w))for i in idxs], axis=0)

    pos = np.stack(np.mgrid[0:h:1, 0:w:1], axis=-1)
    h_t, w_t = translation
    pdf = multivariate_normal.pdf(pos, mean=[h // 2 + h_t, w // 2 + w_t], cov=[[gaussian_var, 0], [0, gaussian_var]])

    Y = np.stack([np.expand_dims(pdf, axis=-1) for b in range(0, batchsize)], axis=0)
    Y /= np.max(Y)

    return np.float32(X), np.float32(Y)


def get_batch_noise(shape, batchsize, gaussian_var, translation):

    h, w, c = shape

    X = np.stack([np.random.uniform(-1, 1, size=(h, w, c)) for b in range(0, batchsize)], axis=0)

    pos = np.stack(np.mgrid[0:h:1, 0:w:1], axis=-1)
    h_t, w_t = translation
    pdf = multivariate_normal.pdf(pos, mean=[h // 2 + h_t, w // 2 + w_t], cov=[[gaussian_var, 0], [0, gaussian_var]])

    Y = np.stack([np.expand_dims(pdf, axis=-1) for b in range(0, batchsize)], axis=0)
    Y /= np.max(Y)

    return np.float32(X), np.float32(Y)


def get_batch_uniform(shape, batchsize, gaussian_var, translation):

    h, w, c = shape

    X = np.stack([np.ones(shape=(h, w, c)) for b in range(0, batchsize)], axis=0)

    pos = np.stack(np.mgrid[0:h:1, 0:w:1], axis=-1)
    h_t, w_t = translation
    pdf = multivariate_normal.pdf(pos, mean=[h // 2 + h_t, w // 2 + w_t], cov=[[gaussian_var, 0], [0, gaussian_var]])

    Y = np.stack([np.expand_dims(pdf, axis=-1) for b in range(0, batchsize)], axis=0)
    Y /= np.max(Y)

    return np.float32(X), np.float32(Y)


def get_batch_uniform_gt_hole(shape, batchsize, side_ext, side_int):

    h, w, c = shape

    X = np.stack([np.ones(shape=(h, w, c)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    Y = np.stack([create_square_with_hole(32, 32, side_ext=side_ext, side_int=side_int) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch(shape, batchsize, gaussian_var, type, translation):

    assert type in ['voc', 'noise', 'dreyeve', 'cityscapes', 'uniform'], \
        'Type must be either `voc`, `noise`, `dreyeve`, `cityscapes`, `uniform`'

    if type == 'voc':
        B = get_batch_voc(shape, batchsize, gaussian_var, translation)
    elif type == 'noise':
        B = get_batch_noise(shape, batchsize, gaussian_var, translation)
    elif type == 'dreyeve':
        B = get_batch_dreyeve(shape, batchsize, gaussian_var, translation)
    elif type == 'cityscapes':
        B = get_batch_cityscapes(shape, batchsize, gaussian_var, translation)
    else:
        B = get_batch_uniform(shape, batchsize, gaussian_var, translation)

    return B

if __name__ == '__main__':
    X_num, Y_num = get_batch_uniform_gt_hole(shape=(128, 128, 3),
                                             batchsize=16,
                                             side_ext=8,
                                             side_int=4)
