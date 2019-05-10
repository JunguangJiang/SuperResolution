"""
Data Pre-processing

preprocessing.py does the following work:
    1. extract the zip files
    2. convert y4m file to bmp files

Usage:

"""
import os
import zipfile
from tqdm import tqdm
import shutil


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
    convert a y4m_file to bmp files into a certain directory
    :param y4m_dir: the dir name of the y4m file
    :param y4m_file: the file name of the y4m file
    :param bmp_files_dir: the directory where the bmp files should be placed
    :return:
    """
    pattern = os.path.join(bmp_files_dir, y4m_file.split(".")[0] + "_l%3d.bmp")
    os.system("ffmpeg -i {} -vsync 0 {} -y".format(os.path.join(y4m_dir, y4m_file), pattern))


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
    make_data_set("data", "new_data")
