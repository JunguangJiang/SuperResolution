"""
Data Pre-processing

Usage:

1. Given a json file, then download the zip file (the OS should install wget first)
    python preprocess.py download [json_file] [target_dir]

2. Extract the zip files and convert y4m file to bmp files (the OS should install ffmpeg first)
    python preprocess.py extract [source_dir] [target_dir]

"""
import os
import zipfile
from tqdm import tqdm
import shutil
import json
import sys


def unzip(filename, new_dir):
    """
    unzip a zip file to a certain directory
    :param filename: the name of the zip file
    :param new_dir: the directory where the unzip files should be placed
    :return:
    """
    zip_file = zipfile.ZipFile(filename)
    for name in tqdm(zip_file.namelist()):
        zip_file.extract(name, new_dir)
    zip_file.close()


def y4m_2_bmp(y4m_dir, y4m_file, bmp_files_dir):
    """
    convert a y4m file to bmp files into a certain directory
    :param y4m_dir: the dir name of the y4m file
    :param y4m_file: the file name of the y4m file
    :param bmp_files_dir: the directory where the bmp files should be placed
    :return:
    """
    pattern = os.path.join(bmp_files_dir, y4m_file.split(".")[0] + "_%3d.bmp")
    os.system("ffmpeg -i {} -vsync 0 {} -y".format(os.path.join(y4m_dir, y4m_file), pattern))


def bmp_2_y4m(bmp_files_dir, bmp_file_prefix, y4m_dir):
    """
    convert bmp files to a y4m file
    :param bmp_files_dir: the dir name of the bmp files
    :param bmp_file_prefix: the prefix of the bmp file, such as `Youku_00099_l_l`
    :param y4m_dir: the directory where the y4m file should be placed
    :return:
    """
    os.system("ffmpeg -i {}_%3d.bmp  -pix_fmt yuv420p  -vsync 0 {}.y4m -y".format(os.path.join(bmp_files_dir, bmp_file_prefix), os.path.join(y4m_dir, bmp_file_prefix)))


def copy_dir_structure(source_root, target_root):
    """
    copy the directory structure without copying the file
    :param source_root: the name of the source root dir
    :param target_root: the name of the target root dir
    :return:
    """
    if not os.path.isdir(source_root):
        print("Not found source directory: {}".format(source_root))
        return False
    os.makedirs(target_root, exist_ok=True)

    for path, dir_list, file_list in os.walk(source_root):
        new_dir = path.replace(source_root, target_root)
        os.makedirs(new_dir, exist_ok=True)  # Ignore the "Dir already exists" Error
    return True


def download_origin_data(json_file, root_dir):
    """
    download the zip files to the root_dir
    :param json_file: a json file which has the following content
    ```
    { "train":{
        "input":["", ...],
        "label":["", ...]
    }, ... }
    ```
    :param root_dir: the root dir where the zip files will be placed
    :return:
    """
    os.makedirs(root_dir, exist_ok=True)
    with open(json_file) as f:
        json_obj = json.load(f)
        for dir in json_obj.keys():
            path = os.path.join(root_dir, dir)
            os.makedirs(path, exist_ok=True)
            for sub_dir in json_obj[dir].keys():
                sub_path = os.path.join(path, sub_dir)
                os.makedirs(sub_path, exist_ok=True)
                for url in json_obj[dir][sub_dir]:
                    print("Download {} to {}".format(url, sub_path))
                    os.system("wget -P {} {}".format(sub_path, url))


def make_data_set(source_dir, target_dir):
    """
    construct a data set from source_dir
    :param source_dir: directory where the source zip files are placed
    :param target_dir: directory where the target bmp files are placed
    :return:
    """
    copy_dir_structure(source_dir, target_dir)
    temp_dir = "temp"

    for path, dir_list, file_list in os.walk(source_dir):
        for file_name in file_list:
            if file_name.endswith(".zip"):
                old_file_name = os.path.join(path, file_name)
                new_dir = path.replace(source_dir, target_dir)
                print("extracting {} to {}".format(old_file_name, new_dir))
                os.makedirs(temp_dir, exist_ok=True)
                unzip(old_file_name, temp_dir)
                for file in tqdm(os.listdir(temp_dir)):
                    y4m_2_bmp(temp_dir, file, new_dir)
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
    else:
        if sys.argv[1] == "download":
            json_file = sys.argv[2]
            target_dir = sys.argv[3]
            download_origin_data(json_file, target_dir)
        elif sys.argv[1] == "extract":
            source_dir = sys.argv[2]
            target_dir = sys.argv[3]
            make_data_set(source_dir, target_dir)
        else:
            print(__doc__)

