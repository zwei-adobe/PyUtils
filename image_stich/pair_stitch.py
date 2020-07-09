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
from PIL import ImageFont
from PIL import ImageDraw
import tqdm


src_image_dir = '/home/zwei/Dev/Datasets/tmp_results/Test_Objectness_Result_Det5_4S_Obj_Cls_DBUG'
ref_image_dir = '/home/zwei/Dev/Datasets/tmp_results/TestObj_Result_SOD5_4S_Obj_Cls_0.5_insidemerge'

src_labels = ['Cls']

ref_labels = ['SOD']

def src2ref(src_stem):
    # ref_stem = src_stem[:-4] + '_720'
    ref_stem = src_stem
    return ref_stem

image_labels = src_labels + ref_labels

target_dir = get_dir('Det_SOD')

src_images = get_files_in_dir(src_image_dir)
ref_images = get_files_in_dir(ref_image_dir)


ref_images_dict = {get_stem(x): x for x in ref_images}

font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 32)

for s_src_image_path in tqdm.tqdm(src_images):
    image_stem = get_stem(s_src_image_path)

    if src2ref(image_stem) in ref_images_dict:
        ref_image_path = ref_images_dict[src2ref(image_stem)]
        images = [Image.open(x) for x in [s_src_image_path, ref_image_path]]

        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))
        draw = ImageDraw.Draw(new_im)

        x_offset = 0
        for idx, im in enumerate(images):
            new_im.paste(im, (x_offset, 0))
            draw.text((x_offset+im.size[0]//2, 20), image_labels[idx], fill='yellow', font=font)

            x_offset += im.size[0]

        new_im.save(os.path.join(target_dir, os.path.basename(s_src_image_path)))
    else:
        continue



