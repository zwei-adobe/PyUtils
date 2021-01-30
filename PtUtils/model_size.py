#
# This file is part of the DR_mmsegmentation project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Dec-23.
# 12: 30
# All Rights Reserved
#


def model_parameters(model):
    size_in_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    return size_in_params


def parameters2MB(size_in_params, precision_bit=16.):
    size_in_MB = size_in_params * precision_bit * 1. / (8 * 1024 * 1024)
    return size_in_MB