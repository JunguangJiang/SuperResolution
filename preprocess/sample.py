import os
import random
import glob
import shutil
import tqdm


def _get_slice(origin_list, slice_list):
    new_list = []
    for s in slice_list:
        new_list.append(origin_list[s])
    return new_list


def sample_dataset(src, dst, ratio):
    """
    从数据集src中采样，得到数据集dst，采样比例为ratio
    :param src:
    :param dst:
    :param ratio:
    :return:
    """
    dst_input_dir = os.path.join(dst, "input")
    dst_label_dir = os.path.join(dst, "label")
    if os.path.exists(dst_input_dir):
        shutil.rmtree(dst_input_dir)
    if os.path.exists(dst_label_dir):
        shutil.rmtree(dst_label_dir)
    os.mkdir(dst_input_dir)
    os.mkdir(dst_label_dir)
    input_file_names = sorted(
        glob.glob(os.path.join(src, 'input', '*' + '.bmp'))
    )
    label_file_names = sorted(
        glob.glob(os.path.join(src, 'label', '*' + '.bmp'))
    )
    input_size = len(input_file_names)
    sample_file_rank = random.sample(range(input_size), int(min(1, ratio) * input_size))
    sample_input_file_names = _get_slice(input_file_names, sample_file_rank)
    sample_label_file_names = _get_slice(label_file_names, sample_file_rank)

    for file_name in tqdm.tqdm(sample_input_file_names):
        shutil.copy(file_name, file_name.replace(src, dst))
    for file_name in tqdm.tqdm(sample_label_file_names):
        shutil.copy(file_name, file_name.replace(src, dst))


if __name__ == '__main__':
    # sample_dataset("YoukuDataset/image/train", "YoukuDataset/sample/train", 0.05)
    sample_dataset("YoukuDataset/image2/valid", "YoukuDataset/sample3/valid", 0.1)

