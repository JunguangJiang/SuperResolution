import torch
import torch.nn as nn

from model import common


def params(scale):
    return {
        2: (6, 2, 2),
        4: (8, 4, 2),
        6: (10, 6, 2),
        8: (12, 8, 2)
    }[scale]

def deconv_block(in_channels, hidden, zoomout):
    kernel_size, stride, padding = params(zoomout)

    deconv = nn.ConvTranspose2d(
        in_channels, hidden, kernel_size,
        stride=stride, padding=padding
    )

    return nn.Sequential(deconv, nn.PReLU(hidden))


class LinearUpsampler(nn.Module):

    def __init__(self, in_channels, nr, out_channels, zoomout, upsampler, conv, alpha=0.9):
        super(LinearUpsampler, self).__init__()

        self.deconv_block = deconv_block(in_channels, nr, zoomout)
        self.upsampler = upsampler
        self.conv = conv
        self.alpha = alpha

    def forward(self, x):

        x1 = self.deconv_block(x)
        x = self.upsampler(x)

        x = (1 - self.alpha) * x1 + self.alpha * x
        x = self.conv(x)
        return x

class DeconvBottleneck(nn.Module):

    def __init__(self, in_channels, nr, out_channels, zoomout, zoomin):
        super(DeconvBottleneck, self).__init__()

        self.deconv_block = deconv_block(in_channels, nr, zoomout)

        kernel_size, stride, padding = params(zoomin)
        self.conv = nn.Conv2d(
            nr, out_channels, kernel_size,
            stride=stride, padding=padding
        )

    def forward(self, x):

        x = self.deconv_block(x)
        x = self.conv(x)

        return x