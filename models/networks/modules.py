import torch
import torch.nn as nn
import torch.nn.functional as F


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)

        return x * y


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm=True, leaky=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm=True, leaky=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentiveDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=True):
        super().__init__()
        self.down = Down(in_channels, out_channels, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x):
        return self.attention(self.down(x))


class AttentiveUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm=False, leaky=True):
        super().__init__()
        self.up = Up(in_channels, out_channels, bilinear, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x1, x2):
        return self.attention(self.up(x1, x2))


class AttentiveDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x):
        return self.attention(self.conv(x))
