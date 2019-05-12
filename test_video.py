import os
from preprocess import y4m_2_bmp, bmp_2_y4m
import torch
import tqdm
import data_utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import shutil
import setting


def increase_video_resolution(input_video_dir, input_video_name, output_video_dir, model):
    """
    convert LR videos to HR videos
    :param input_video_dir: the dir name of the input video
    :param input_video_name: the file name of the input video
    :param output_video_dir: the directory where the output video should be placed
    :param model:
    :return:
    """
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    y4m_2_bmp(input_video_dir, input_video_name, temp_dir)

    data_set = data_utils.TestDatasetFromFolder(temp_dir, input_transform=transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.ToTensor()
    ]))
    data_loader = DataLoader(data_set, batch_size=64, num_workers=4, shuffle=False)
    transform = transforms.ToPILImage()
    for input, name in data_loader:
        if torch.cuda.is_available():
            input = input.cuda()
        output = model(input)
        output = output.squeeze(0)
        output = transform(output.squeeze(0))
        output.save(name[0].replace('l', 'h'))

    input_video_prefix = input_video_name.split(".")[0]
    output_video_prefix = input_video_prefix.replace("l", "h")
    bmp_2_y4m(temp_dir, output_video_prefix, output_video_dir)

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    input_video_dir = "data/test/input"
    output_video_dir = "data/test/output"
    video_names = [x for x in os.listdir(input_video_dir) if data_utils.is_y4m_file(x)]
    model = setting.model
    if torch.cuda.is_available():
        model = model.cuda()
    model_name = setting.model_name
    model.load_state_dict(torch.load("model/"+model_name))

    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    for video_name in tqdm.tqdm(video_names, desc='convert LR videos to HR videos'):
        increase_video_resolution(input_video_dir, video_name, output_video_dir, model)

