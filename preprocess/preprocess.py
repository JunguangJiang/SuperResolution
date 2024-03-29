"""
Data Pre-processing

Usage:

1. Given a json file, then download the zip file (the OS should install wget first)
    python preprocess.py download [json_file] [target_dir]

2. Extract the zip files and convert y4m file to bmp files (the OS should install ffmpeg first)
    python preprocess.py zip2bmp [source_dir] [target_dir]

3. Convert all the bmp files in a directory to y4m files in another directory
    python preprocess.py bmp2y4m [source_dir] [target_dir]

4. Convert all the y4m files in a directory to yuv files in another directory
    python preprocess.py y4m2yuv [source_dir] [target_dir]

5. Convert all the y4m files to the final zip files
    python preprocess.py zip [source_dir] [target_dir]

"""
import os
import zipfile
from tqdm import tqdm
import shutil
import json
import sys
import glob
import cv2
import numpy as np



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


def y4ms_2_yuvs(y4m_dir, yuv_dir):
    """convert all the y4m files in y4m_dir to yuv files in yuv_dir"""
    if os.path.exists(yuv_dir):
        shutil.rmtree(yuv_dir)
    os.mkdir(yuv_dir)
    names_sr = sorted(
        glob.glob(os.path.join(y4m_dir, '*.y4m'))
    )
    for name in tqdm(names_sr):
        base_name = os.path.basename(name).replace('.y4m', '.yuv')
        command = "ffmpeg -i {} -vsync 0 {}  -y".format(name, os.path.join(yuv_dir, base_name))
        os.system(command)


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


def zips_2_bmps(source_dir, target_dir):
    """
    construct a YoukuDataSet set from source_dir
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


def bmps_2_y4ms(source_dir, target_dir, ratio=1):
    """
    Convert part of the bmp files in source_dir to y4m files
    :param source_dir:
    :param target_dir: the dir where y4m files should be placed
    :return:
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    names_sr = sorted(
        glob.glob(os.path.join(source_dir, '*_001.bmp'))
    )
    names_sr = names_sr[0: int(ratio * len(names_sr))]
    for name in tqdm(names_sr):
        bmp_2_y4m(source_dir, os.path.basename(name).strip("_001.bmp"), target_dir)


def create_final_result(source_dir, target_dir, ratio=0.1):
    """
    sample from the source_dir and create the final y4m files in target_dir
    :param source_dir:
    :param target_dir:
    :param ratio: top ratio will be reserved, while others will be sub-sample by 1/25.
    :return:
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    names_sr = sorted(
        glob.glob(os.path.join(source_dir, '*.y4m'))
    )
    total_size = len(names_sr)
    reserved_size = int(total_size * ratio)
    for i in tqdm(range(reserved_size)):
        shutil.copy(names_sr[i], names_sr[i].replace(source_dir, target_dir))
    for i in tqdm(range(reserved_size, total_size)):
        os.system("ffmpeg -i {} -vf select='not(mod(n\,25))' -vsync 0  -y {}".format(
            names_sr[i],
            names_sr[i].replace(source_dir, target_dir).replace("_h_Res.y4m", "_h_Sub25_Res.y4m")
        ))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
    else:
        if sys.argv[1] == "download":
            json_file = sys.argv[2]
            target_dir = sys.argv[3]
            download_origin_data(json_file, target_dir)
        else:
            source_dir = sys.argv[2]
            target_dir = sys.argv[3]
            if sys.argv[1] == "zip2bmp":
                zips_2_bmps(source_dir, target_dir)
            elif sys.argv[1] == "bmp2y4m":
                bmps_2_y4ms(source_dir, target_dir)
            elif sys.argv[1] == 'y4m2yuv':
                y4ms_2_yuvs(source_dir, target_dir)
            elif sys.argv[1] == "zip":
                create_final_result(source_dir, target_dir)
            else:
                print(__doc__)

