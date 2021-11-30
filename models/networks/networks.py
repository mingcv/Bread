from models.networks.modules import *


class BaseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(BaseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class IAN(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(IAN, self).__init__(in_channels, out_channels, norm)


class ANSN(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(ANSN, self).__init__(in_channels, out_channels, norm)
        self.outc = OutConv(32, out_channels, act=False)


class FuseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=False):
        super(FuseNet, self).__init__()
        self.inc = AttentiveDoubleConv(in_channels, 32, norm=norm, leaky=False)
        self.down1 = AttentiveDown(32, 64, norm=norm, leaky=False)
        self.down2 = AttentiveDown(64, 64, norm=norm, leaky=False)
        self.up1 = AttentiveUp(128, 32, bilinear=True, norm=norm, leaky=False)
        self.up2 = AttentiveUp(64, 32, bilinear=True, norm=norm, leaky=False)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    for key in FuseNet(4, 2).state_dict().keys():
        print(key)
