from torch.utils.data.dataset import Dataset
from PIL import Image
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.bmp'])


def is_y4m_file(filename):
    return filename.endswith('.y4m')


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = os.path.join(dataset_dir, "input")
        self.target_dir = os.path.join(dataset_dir, "label")
        self.input_filenames = [os.path.join(self.image_dir, x) for x in
                                os.listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [os.path.join(self.target_dir, x) for x in
                                 os.listdir(self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.open(self.input_filenames[index])
        target = Image.open(self.target_filenames[index])
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.input_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, image_dir, input_transform=None):
        super(TestDatasetFromFolder, self).__init__()
        self.input_filenames = [os.path.join(image_dir, x) for x in
                                os.listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = Image.open(self.input_filenames[index])
        if self.input_transform:
            image = self.input_transform(image)

        return image, self.input_filenames[index]

    def __len__(self):
        return len(self.input_filenames)

