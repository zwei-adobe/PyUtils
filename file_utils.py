import os
import os.path as osp
import datetime
import glob

def get_dir(directory, isFile=False):
    """
    Creates the given directory if it does not exist.
    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if isFile:
        if not os.path.exists(os.path.dirname(directory)):
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


def relative_path2root(full_path, root_path):
    start_id = len(root_path) + 1 if len(root_path) > 0 else 0
    return full_path[start_id:]