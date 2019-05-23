import torch.nn as nn
import torch.nn.functional as F
import torch


class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        input = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear')
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        return x + input


if __name__ == "__main__":
    model = ESPCN(upscale_factor=3)
    print(model)
