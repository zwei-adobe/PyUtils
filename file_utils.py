import os
import os.path as osp
import datetime
import glob
from pathlib import Path

def get_parent_directory(cur_file_path, parent_level=1, to_str=True):
    cur_dir = Path(osp.dirname(osp.abspath(osp.expanduser(cur_file_path))))
    for i in range(parent_level):
        cur_dir = cur_dir.parent
    if to_str:
        return str(cur_dir)
    else:
        return cur_dir

def keep_cv_supported_image_formats(file_list, formats=None):
    """
    Filter a list of files to supported images formats
    Args:
        file_list:
        formats: list, if none, will use default opencv supported formats:
        https://docs.opencv.org/master/d4/da8/group__imgcodecs.html

    Returns:

    """
    if formats is None:
        formats = ['bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'jp2', 'png', 'webp',  'pbm', 'pgm', 'ppm', 'pxm', 'pnm' ,
                   'pfm', 'tiff', 'tif', 'pic', 'hdr', 'exr']
    formats = set(formats)
    image_list = []

    for s_file_path in file_list:
        s_file_extension = get_extension(s_file_path)
        if s_file_extension.lower() in formats:
            image_list.append(s_file_path)
        else:
            continue

    return image_list



def get_dir(directory, isFile=False):
    """
    Creates the given directory if it does not exist.
    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    # isFile = os.path.isfile(directory)
    if isFile:
        # fix potential bug with directory is a file without name
        if len(os.path.dirname(directory))>0 and  not os.path.exists(os.path.dirname(directory)):
            os.makedirs(os.path.dirname(directory))
        return directory
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

def get_file_dir(file_path):
    directory = os.path.dirname(file_path)
    get_dir(directory)
    return file_path

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')[:-7].replace('-', '')

def get_image_list(images):
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    return imlist


def get_subdir_imagelist(directory):
    image_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.jpg':
                image_list.append(os.path.join(path, name))
    return image_list


def get_stem(file_path, keep_path=False):
    basename = os.path.basename(file_path)
    stem = os.path.splitext(basename)[0]
    if keep_path:
        return os.path.join(os.path.dirname(file_path), stem)
    return stem

def get_extension(file_path):
    basename = os.path.basename(file_path)
    ext = basename.split('.')[-1]
    return ext


def add_suffix(file_path, suffix='1'):
    stem = get_stem(file_path, keep_path=True)
    ext = get_extension(file_path)
    return '{}_{}.{}'.format(stem, suffix, ext)

def get_files_in_dir(target_directory, formatter=None, ALL=True):
    #zwtodo: this is actually a bug report, all files in the first directory would not be recorded!
    file_list = glob.glob(os.path.join(target_directory, "**/*"), recursive=ALL)
    if formatter is None:
        file_list = [x for x in file_list if os.path.isfile(x)]
    else:
        file_list = [x for x in file_list if os.path.isfile(x) and get_extension(x) == formatter]

    return file_list


def get_extensions(file_list):
    extensions = set()
    for s_file_path in file_list:
        extensions.add(os.path.splitext(os.path.basename(s_file_path))[1])
    return list(extensions)


def update_path(original_path, cur_directory, sub_dir):
    path_parts = original_path.split(os.sep)
    new_path = os.path.join(cur_directory, *path_parts[sub_dir:])
    return new_path

def keep_last(original_path, keep_last_n=1):
    path_parts = original_path.split(os.sep)
    n_parts = len(path_parts)
    assert n_parts>=keep_last_n, '{} is shorter than {}'.format(original_path, keep_last_n)
    new_path = os.path.join(*path_parts[n_parts-keep_last_n:])
    return new_path

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# def relative_path2root(full_path, root_path):
#     start_id = len(root_path) + 1 if len(root_path) > 0 else 0
#     return full_path[start_id:]

def relative_path2root(full_path, root_path):
    #zwtodo: this needs more tests
    start_id = len(root_path) + 1 if len(root_path) > 0 else 0
    if root_path == '/':
        start_id=1
    return full_path[start_id:]


def relative_path2root_v2(full_path, root_path='/'):
    # this is a better implementation, slower, but more robust
    full_path_parts = full_path.split(os.sep)
    root_path_parts = root_path.split(os.sep)
    rel_path_parts = []
    for idx, s_part in enumerate(full_path_parts):
        if idx < len(root_path_parts):
            assert s_part == root_path_parts[idx], 'file path doesnt match: abs: {} vs root: {}'.format(full_path, root_path)
            continue
        rel_path_parts.append(s_part)
    return os.sep.join(rel_path_parts)