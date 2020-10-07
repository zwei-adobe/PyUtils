#
# This file is part of the zw_detect project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-07.
# 11: 32
# All Rights Reserved
#
import numpy as np

#zwtodo: these operations should not be changing the input values

def extend_box_by_ratio(box, image_size_xy, extend_factor=0.1, box_type='xywh', convert2int=False):

    assert box_type in {'xyxy', 'xywh'}
    assert isinstance(image_size_xy, (list, tuple))
    if isinstance(extend_factor, (list, tuple)):
        assert len(extend_factor) == 2
        extend_factor_x, extend_factor_y = extend_factor
    else:
        extend_factor_x = extend_factor_y = extend_factor

    if box_type == 'xyxy':
        box = convert_list_xyxy2xywh(box)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    extend_w = w * extend_factor_x // 2
    extend_h = h * extend_factor_y // 2

    new_x0 = max(x - extend_w, 0)
    new_y0 = max(y - extend_h, 0)
    new_x1 = min(new_x0 + w + extend_w * 2, image_size_xy[0])
    new_y1 = min(new_y0 + h + extend_h * 2, image_size_xy[1])
    new_box = [new_x0, new_y0, new_x1, new_y1]

    if box_type == 'xywh':
        new_box = convert_list_xyxy2xywh(new_box)

    if convert2int:
        new_box = list(map(int, new_box))
    return new_box

def extend_box_by_pixel(box, image_size_xy=None, extend_pixel=50, box_type='xywh', convert2int=False):
    assert box_type in {'xyxy', 'xywh'}
    assert isinstance(image_size_xy, (list, tuple))
    if isinstance(extend_pixel, (list, tuple)):
        assert len(extend_pixel) == 2
        extend_w, extend_h = extend_pixel
    else:
        extend_w = extend_h = extend_pixel

    if box_type == 'xyxy':
        box = convert_list_xyxy2xywh(box)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    new_x0 = max(x - extend_w, 0)
    new_y0 = max(y - extend_h, 0)
    new_x1 = min(new_x0 + w + extend_w * 2, image_size_xy[0]) if image_size_xy is not None else new_x0 + w + extend_w * 2
    new_y1 = min(new_y0 + h + extend_h * 2, image_size_xy[1]) if image_size_xy is not None else new_y0 + h + extend_h * 2
    new_box = [new_x0, new_y0, new_x1, new_y1]

    if box_type == 'xywh':
        new_box = convert_list_xyxy2xywh(new_box)

    if convert2int:
        new_box = list(map(int, new_box))
    return new_box


def convert_list_xywh2xyxy(xywh, convert2int=False):
    if isinstance(xywh[0], list):
        xyxy = [convert_list_xywh2xyxy(m) for m in xywh]
        return xyxy
    else:
        xyxy = [xywh[0], xywh[1], xywh[2] + xywh[0], xywh[3]+xywh[1]]
        if convert2int:
            xyxy = list(map(int, xyxy))
        return xyxy

def convert_list_xyxy2xywh(xyxy, convert2int=False):
    if isinstance(xyxy[0], list):
        xywh = [convert_list_xyxy2xywh(m) for m in xyxy]
        return xywh
    else:
        xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        if convert2int:
            xywh = list(map(int, xywh))
        return xywh


def sort_bboxes_xywh(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    return idxs


def _convert_boxes(boxes):
    # convert boxes to np.array form
    assert isinstance(boxes, (list, np.ndarray,))
    return np.asarray(boxes)


def non_max_suppression_fast(boxes, overlapThresh):
    """
    the boxes are already sorted by areas from low to high
    :param boxes: x, y, w, h and already sorted by ares
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of compute_coverage_masking
        overlap = (w * h) / (area[idxs[:last]] + area[idxs[last]] - w * h )
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick


def remove_inside_boxes(boxes, boarder=5):
    # should be sorted by regions from high to low
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    # if boxes.dtype.kind == "i":
    #     boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(int(i))
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        # xx1 = np.maximum(x1[i], x1[idxs[:last]])
        # yy1 = np.maximum(y1[i], y1[idxs[:last]])
        # xx2 = np.minimum(x2[i], x2[idxs[:last]])
        # yy2 = np.minimum(y2[i], y2[idxs[:last]])
        xx1_inside = x1[i] <= x1[idxs[:last]] + boarder
        yy1_insdie = y1[i] <= y1[idxs[:last]] + boarder
        xx2_inside = x2[i] >= x2[idxs[:last]] - boarder
        yy2_inside = y2[i] >= y2[idxs[:last]] - boarder
        total_inside = np.where(xx1_inside & yy1_insdie & xx2_inside & yy2_inside)
        # compute the width and height of the bounding box
        #     w = np.maximum(0, xx2 - xx1 + 1)
        #     h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of compute_coverage_masking
        # compute_coverage_masking = (w * h) / (area[idxs[:last]] + area[idxs[last]] - w * h )
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               total_inside[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], pick