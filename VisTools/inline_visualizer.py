#
# This file is part of the DR_Segmentation project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Nov-11.
# 07: 39
# All Rights Reserved
#
import matplotlib.pyplot as plt

def visualize(save_path=None, **images):
    """PLot images in one row."""
    n = len(images)
    fig = plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)