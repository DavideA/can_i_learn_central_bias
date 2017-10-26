import numpy as np


def create_square_filled(height, width, square_side):
    gt = np.zeros(shape=(height, width, 1), dtype=np.float32)

    assert square_side > 0

    center_h, center_w = height // 2, width // 2

    square_start_h = center_h - square_side // 2
    square_end_h   = center_h + square_side // 2
    square_start_w = center_w - square_side // 2
    square_end_w   = center_w + square_side // 2
    gt[square_start_h: square_end_h, square_start_w: square_end_w, :] = 1.

    return gt


def create_square_with_hole(height, width, side_ext, side_int):
    gt = np.zeros(shape=(height, width, 1), dtype=np.float32)

    assert side_int < side_ext

    center_h, center_w = height // 2, width // 2

    ext_square_start_h = center_h - side_ext // 2
    ext_square_end_h   = center_h + side_ext // 2
    ext_square_start_w = center_w - side_ext // 2
    ext_square_end_w   = center_w + side_ext // 2
    gt[ext_square_start_h: ext_square_end_h, ext_square_start_w: ext_square_end_w, :] = 1.

    int_square_start_h = center_h - side_int // 2
    int_square_end_h   = center_h + side_int // 2
    int_square_start_w = center_w - side_int // 2
    int_square_end_w   = center_w + side_int // 2
    gt[int_square_start_h: int_square_end_h, int_square_start_w: int_square_end_w, :] = 0.

    return gt


def get_batch_uniform_gt_hole(shape, batchsize, side_ext, side_int):

    h, w, c = shape

    X = np.stack([np.ones(shape=(h, w, c)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    Y = np.stack([create_square_with_hole(32, 32, side_ext=side_ext, side_int=side_int) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_uniform_gt_filled(shape, batchsize, square_side):

    h, w, c = shape

    X = np.stack([np.ones(shape=(h, w, c)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    Y = np.stack([create_square_filled(32, 32, square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


def get_batch_uniform_gt_filled_x_bias(shape, batchsize, square_side):

    h, w, c = shape
    X = np.random.rand(batchsize, h, w, c)
    # X = np.stack([np.ones(shape=(h, w, c)) for b in range(0, batchsize)], axis=0)
    X -= 0.5

    # Add a bias in the input
    # Bias kind: two lines
    # for x in X:
    #     x[:, 32] = 0.25
    #     x[32, :] = 0.1
    # Bias kind: little square in top left corner
    for x in X:
        x[20:30, 20:30] = 0.25

    Y = np.stack([create_square_filled(32, 32, square_side=square_side) for b in range(0, batchsize)], axis=0)

    return np.float32(X), np.float32(Y)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    X_num, Y_num = get_batch_uniform_gt_filled(shape=(128, 128, 3),
                                               batchsize=16,
                                               square_side=8)

    for y in Y_num:
        plt.imshow(np.squeeze(y))
        plt.show()
