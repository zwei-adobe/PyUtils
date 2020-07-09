#
# This file is part of the SOD_dp project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-24.
# 17: 48
# All Rights Reserved
#

from PyUtils.file_utils import *
from PIL import Image
import tqdm
#zwadd: not done yet, should be able to show multiple images
# raise NotImplementedError

src_image_dir = '/home/zwei/Dev/Datasets/tmp_results/Test_Objectness_Result_JasonObjectness'
ref_image_dirs = ['/home/zwei/Dev/Datasets/tmp_results/Test_Objectness_Result_Det5_4S_Obj_Cls',
                 '/home/zwei/Dev/Datasets/tmp_results/Test_Objectness_Result_Det5_4S_Obj_Cls_DBUG']


target_dir = get_dir('3stitch')

src_images = get_files_in_dir(src_image_dir)
ref_images_list = [get_files_in_dir(ref_image_dir) for ref_image_dir in ref_image_dirs]


ref_images_dicts = [{get_stem(x): x for x in ref_images} for ref_images in ref_images_list]


for s_src_image_path in tqdm.tqdm(src_images):
    image_stem = get_stem(s_src_image_path)

    ref_image_paths = []
    for s_dict in ref_images_dicts:
        if get_stem(s_src_image_path) in s_dict:
            ref_image_path = s_dict[get_stem(s_src_image_path)]
            ref_image_paths.append(ref_image_path)

    if len(ref_image_paths) != len(ref_images_dicts):
        continue

    ref_image_paths.insert(0, s_src_image_path)
    images = [Image.open(x) for x in ref_image_paths]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(os.path.join(target_dir, os.path.basename(s_src_image_path)))
# else:
#     continue



