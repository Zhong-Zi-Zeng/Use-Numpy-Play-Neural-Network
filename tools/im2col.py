import numpy as np

def im2col(images, flt_h, flt_w, stride=1, pad=0):
    images_pad = np.pad(images, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    batch, channel, img_height, img_width = images_pad.shape

    out_h = (img_height - flt_h + 2 * pad) // stride + 1
    out_w = (img_width - flt_w + 2 * pad) // stride + 1

    metric_height = channel * flt_h * flt_w * batch
    metric_width = out_h * out_w

    result = np.zeros((metric_height, metric_width))
    col_idx = 0

    for row in range(0, img_height - flt_h + 1, stride):
        for col in range(0, img_width - flt_w + 1, stride):
            result[:, col_idx] = images_pad[:, :, row: row + flt_h, col: col + flt_w].reshape(-1, )
            col_idx += 1

    result = np.reshape(result, (batch, channel * flt_h * flt_w, out_h * out_w))
    result = np.concatenate(result, axis=1)

    return result




