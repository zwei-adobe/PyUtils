#
# This file is part of the SOD_dp project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-May-21.
# 11: 29
# All Rights Reserved
# modified from facebook

import colorsys
import logging
import math
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
# import pycocotools.mask as mask_util
# import torch
# from fvcore.common.file_io import PathManager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageOps
# from fvcore.common.file_io import PathManager
from .fb_PathManager import PathManager
# from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
import warnings
# warnings.filterwarnings("error")
from .colormap import random_color

logger = logging.getLogger(__name__)

__all__ = ["ColorMode", "VisImage", "Visualizer"]


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """





def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            # faster than matplotlib's imshow
            cv2.imwrite(filepath, self.get_image()[:, :, ::-1])
        else:
            # support general formats (e.g. pdf)
            self.ax.imshow(self.img, interpolation="nearest")
            self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = cv2.resize(self.img, (width, height))
        else:
            img = self.img

        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        # imshow is slow. blend manually (still quite slow)
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")

        return visualized_image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            print('{} problematic, pass'.format(file_name))
            return None
            # pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


class Visualizer:
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (MetadataCatalog): image metadata.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        # self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 100, 8 // scale
        )
        self._instance_mode = instance_mode


    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def draw_boxes_xyxy_with_labels(self, input_bboxes, labels):
        boxes = input_bboxes
        colors = None
        alpha = 0.5

        self.overlay_xyxy_boxes(
            boxes=boxes,
            labels=labels,
            masks=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_boxes_xyxy_with_nonoverlap_labels(self, input_bboxes, labels):
        boxes = input_bboxes
        colors = None
        alpha = 0.5

        self.overlay_xyxy_boxes_non_overlap(
            boxes=boxes,
            labels=labels,
            masks=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_boxes_xyxy(self, input_bboxes, scores=None):
        boxes = input_bboxes
        labels = ['{:.2f}'.format(s_score) for s_score in scores] if scores else None
        colors = None
        alpha = 0.5

        self.overlay_xyxy_boxes(
            boxes=boxes,
            labels=labels,
            masks=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_boxes_xywh_with_labels(self, input_bboxes, labels):
        boxes = input_bboxes
        colors = None
        alpha = 0.5
        boxes = Visualizer._convert_boxes(boxes)
        boxes_xyxy = Visualizer.boxes_xywh2xyxy(boxes)

        self.overlay_xyxy_boxes(
            boxes=boxes_xyxy,
            labels=labels,
            masks=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


    def draw_boxes_xywh(self, input_bboxes, scores=None):
        boxes = input_bboxes
        labels = ['{:.2f}'.format(s_score) for s_score in scores] if scores else None
        colors = None
        alpha = 0.5
        boxes = Visualizer._convert_boxes(boxes)
        boxes_xyxy = Visualizer.boxes_xywh2xyxy(boxes)

        self.overlay_xyxy_boxes(
            boxes=boxes_xyxy,
            labels=labels,
            masks=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output



    @staticmethod
    def boxes_xywh2xyxy(boxes):
            Visualizer._convert_boxes(boxes)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            return boxes


    @staticmethod
    def _convert_boxes(boxes):
        # convert boxes to np.array form
        assert isinstance(boxes, (list, np.ndarray,))
        return np.asarray(boxes)





    def overlay_xyxy_boxes(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.

            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)


        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]

        if num_instances == 0:
            return self.output


        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)



            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bboxes()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )



        return self.output

    def overlay_single_xyxy_box(
        self,
        *,
        single_box=None,
        single_label=None,
        assigned_color=None,
        align='left'
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.

            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """

        # num_instances = None
        # if boxes is not None:
        #     boxes = self._convert_boxes(boxes)
        #     num_instances = len(boxes)
        #
        #
        # if labels is not None:
        #     assert len(labels) == num_instances
        # if assigned_colors is None:
        #     assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]

        # if num_instances == 0:
        #     return self.output


        # Display in largest to smallest order to reduce occlusion.
        # areas = None
        # if boxes is not None:
        #     areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        # if areas is not None:
        #     sorted_idxs = np.argsort(-areas).tolist()
        #     # Re-order overlapped instances in descending order.
        #     boxes = boxes[sorted_idxs] if boxes is not None else None
        #     labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        #     masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
        #     assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        # for i in range(num_instances):
        #     color = assigned_colors[i]
        if single_box is not None:
            self.draw_box(single_box, edge_color=assigned_color)



            if single_label is not None:
                # first get a box
                if single_box is not None:
                    x0, y0, x1, y1 = 0+5, 0+5, self.output.width, self.output.height
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = align or "left"
                    # elif masks is not None:
                    # x0, y0, x1, y1 = masks[i].bboxes()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    # text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    # horiz_align = "center"
                # else:
                #     continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                    instance_area = (y1 - y0) * (x1 - x0)
                    if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                    ):
                        if y1 >= self.output.height - 5:
                            text_pos = (x1, y0)
                        else:
                            text_pos = (x0, y1)

                    height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                    lighter_color = self._change_color_brightness(assigned_color, brightness_factor=0.7)
                    font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size * 1.2
                    )
                    self.draw_text(
                        single_label,
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                    )



        return self.output


    @staticmethod
    def pointInRect(point, rect, close=5):
        x1, y1, x2, y2 = rect
        # x2, y2 = x1 + w, y1 + h
        x, y = point
        if (x1  <= x + close and x  <= x2 + close):
            if (y1 <= y + close and y  <= y2+close):
                return True
        return False


    def overlay_xyxy_boxes_non_overlap(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.

            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)


        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]

        if num_instances == 0:
            return self.output


        # Display from smallest to largest.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_idxs = np.argsort(areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        marked_rects = []
        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)



            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bboxes()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )


                for tmp_anchor in [(x0,  y0), (x1, y0), (x0, y1), (x1, y1)]:
                    for s_rect in marked_rects:
                        if self.pointInRect(tmp_anchor, s_rect):
                            break
                    text_pos = tmp_anchor
                    # point_change = True
                    break

                marked_rects.append(boxes[i])
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )



        return self.output


    """
    Primitive drawing functions:
    """

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output


    # def draw_binary_mask(
    #     self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=4096
    # ):
    #     """
    #     Args:
    #         binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
    #             W is the image width. Each value in the array is either a 0 or 1 value of uint8
    #             type.
    #         color: color of the mask. Refer to `matplotlib.colors` for a full list of
    #             formats that are accepted. If None, will pick a random color.
    #         edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
    #             full list of formats that are accepted.
    #         text (str): if None, will be drawn in the object's center of mass.
    #         alpha (float): blending efficient. Smaller values lead to more transparent masks.
    #         area_threshold (float): a connected component small than this will not be shown.
    #
    #     Returns:
    #         output (VisImage): image object with mask drawn.
    #     """
    #     if color is None:
    #         color = random_color(rgb=True, maximum=1)
    #     if area_threshold is None:
    #         area_threshold = 4096
    #
    #     has_valid_segment = False
    #     binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
    #     mask = GenericMask(binary_mask, self.output.height, self.output.width)
    #     shape2d = (binary_mask.shape[0], binary_mask.shape[1])
    #
    #     if not mask.has_holes:
    #         # draw polygons for regular masks
    #         for segment in mask.polygons:
    #             area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
    #             if area < area_threshold:
    #                 continue
    #             has_valid_segment = True
    #             segment = segment.reshape(-1, 2)
    #             self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
    #     else:
    #         rgba = np.zeros(shape2d + (4,), dtype="float32")
    #         rgba[:, :, :3] = color
    #         rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
    #         has_valid_segment = True
    #         self.output.ax.imshow(rgba)
    #
    #     if text is not None and has_valid_segment:
    #         # TODO sometimes drawn on wrong objects. the heuristics here can improve.
    #         lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
    #         _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
    #         largest_component_id = np.argmax(stats[1:, -1]) + 1
    #
    #         # draw text on the largest component, as well as other very large components.
    #         for cid in range(1, _num_cc):
    #             if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
    #                 # median is more stable than centroid
    #                 # center = centroids[largest_component_id]
    #                 center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
    #                 self.draw_text(text, center, color=lighter_color)
    #     return self.output
    #
    # def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
    #     """
    #     Args:
    #         segment: numpy array of shape Nx2, containing all the points in the polygon.
    #         color: color of the polygon. Refer to `matplotlib.colors` for a full list of
    #             formats that are accepted.
    #         edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
    #             full list of formats that are accepted. If not provided, a darker shade
    #             of the polygon color will be used instead.
    #         alpha (float): blending efficient. Smaller values lead to more transparent masks.
    #
    #     Returns:
    #         output (VisImage): image object with polygon drawn.
    #     """
    #     if edge_color is None:
    #         # make edge color darker than the polygon color
    #         if alpha > 0.8:
    #             edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
    #         else:
    #             edge_color = color
    #     edge_color = mplc.to_rgb(edge_color) + (1,)
    #
    #     polygon = mpl.patches.Polygon(
    #         segment,
    #         fill=True,
    #         facecolor=mplc.to_rgb(color) + (alpha,),
    #         edgecolor=edge_color,
    #         linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
    #     )
    #     self.output.ax.add_patch(polygon)
    #     return self.output

    """
    Internal methods:
    """
    #
    # def _jitter(self, color):
    #     """
    #     Randomly modifies given color to produce a slightly different color than the color given.
    #
    #     Args:
    #         color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
    #             picked. The values in the list are in the [0.0, 1.0] range.
    #
    #     Returns:
    #         jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
    #             color after being jittered. The values in the list are in the [0.0, 1.0] range.
    #     """
    #     color = mplc.to_rgb(color)
    #     vec = np.random.rand(3)
    #     # better to do it in another color space
    #     vec = vec / np.linalg.norm(vec) * 0.5
    #     res = np.clip(vec + color, 0, 1)
    #     return tuple(res)
    #
    # def _create_grayscale_image(self, mask=None):
    #     """
    #     Create a grayscale version of the original image.
    #     The colors in masked area, if given, will be kept.
    #     """
    #     img_bw = self.img.astype("f4").mean(axis=2)
    #     img_bw = np.stack([img_bw] * 3, axis=2)
    #     if mask is not None:
    #         img_bw[mask] = self.img[mask]
    #     return img_bw
    #
    # def _change_color_brightness(self, color, brightness_factor):
    #     """
    #     Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    #     less or more saturation than the original color.
    #
    #     Args:
    #         color: color of the polygon. Refer to `matplotlib.colors` for a full list of
    #             formats that are accepted.
    #         brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
    #             0 will correspond to no change, a factor in [-1.0, 0) range will result in
    #             a darker color and a factor in (0, 1.0] range will result in a lighter color.
    #
    #     Returns:
    #         modified_color (tuple[double]): a tuple containing the RGB values of the
    #             modified color. Each value in the tuple is in the [0.0, 1.0] range.
    #     """
    #     assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    #     color = mplc.to_rgb(color)
    #     polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    #     modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    #     modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    #     modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    #     modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    #     return modified_color
    #
    # def _convert_boxes(self, boxes):
    #     """
    #     Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
    #     """
    #     if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
    #         return boxes.tensor.numpy()
    #     else:
    #         return np.asarray(boxes)
    #
    # def _convert_masks(self, masks_or_polygons):
    #     """
    #     Convert different format of masks or polygons to a tuple of masks and polygons.
    #
    #     Returns:
    #         list[GenericMask]:
    #     """
    #
    #     m = masks_or_polygons
    #     if isinstance(m, PolygonMasks):
    #         m = m.polygons
    #     if isinstance(m, BitMasks):
    #         m = m.tensor.numpy()
    #     if isinstance(m, torch.Tensor):
    #         m = m.numpy()
    #     ret = []
    #     for x in m:
    #         if isinstance(x, GenericMask):
    #             ret.append(x)
    #         else:
    #             ret.append(GenericMask(x, self.output.height, self.output.width))
    #     return ret
    #
    # def _convert_keypoints(self, keypoints):
    #     if isinstance(keypoints, Keypoints):
    #         keypoints = keypoints.tensor
    #     keypoints = np.asarray(keypoints)
    #     return keypoints
    #
    # def get_output(self):
    #     """
    #     Returns:
    #         output (VisImage): the image output containing the visualizations added
    #         to the image.
    #     """
    #     return self.output
