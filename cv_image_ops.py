#
# This file is part of the SVD_SS_v2 project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Oct-19.
# 11: 21
# All Rights Reserved
#

import cv2
import numpy as np


def resize2_min_size_X(image, min_size=1000):
    image_shape = list(image.shape)
    argmin = np.argmin(image_shape[0:2])
    size_factor = min_size*1. / image_shape[argmin]
    image_shape = np.round(np.array(image_shape).astype(float) * size_factor).astype(int).tolist()

    image = cv2.resize(image, (image_shape[1], image_shape[0]))
    return image