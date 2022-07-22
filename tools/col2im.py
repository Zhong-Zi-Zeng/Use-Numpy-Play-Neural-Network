import numpy as np

def col2im(metric, img_shape, flt_h, flt_w, stride=1):
    batch, channel, img_height, img_width = img_shape
    sp_metric = np.array(np.split(metric, batch, axis=1))

    result = np.zeros(img_shape)
    col_idx = 0
    for row in range(0, img_height - flt_h + 1, stride):
        for col in range(0, img_width - flt_w + 1, stride):
            result[:, :, row: row + flt_h, col: col + flt_w] += sp_metric[:, :, col_idx].reshape(batch, channel, flt_h, flt_w)

            col_idx += 1

    return result


