import torch
from torch import nn


class MCIM(nn.Module):
    def __init__(self, in_channels):
        super(MCIM, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.up_1 = nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=2, stride=2)

        self.a_conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.r_conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.r_conv_2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.r_conv_3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.r_conv_4 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7)

        self.softmax = nn.Sigmoid()

        self.conv_2 = nn.Conv3d(in_channels * 4, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(in_channels)

    def forward(self, x, y):
        x1 = self.up_1(y)
        x2 = x + x1
        x2 = self.conv_1(x2)

        x3_1 = self.a_conv_1(x2)
        x3_1 = self.r_conv_1(x3_1)
        x3_1 = self.softmax(x3_1)
        x3_1 = x3_1 * x2

        x3_2 = self.a_conv_1(x2)
        x3_2 = self.r_conv_2(x3_2)
        x3_2 = self.softmax(x3_2)
        x3_2 = x3_2 * x2

        x3_3 = self.a_conv_1(x2)
        x3_3 = self.r_conv_3(x3_3)
        x3_3 = self.softmax(x3_3)
        x3_3 = x3_3 * x2

        x3_4 = self.a_conv_1(x2)
        x3_4 = self.r_conv_4(x3_4)
        x3_4 = self.softmax(x3_4)
        x3_4 = x3_4 * x2

        y1 = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)
        y = self.conv_2(y1)

        return y

class MCIM1(nn.Module):
    def __init__(self, in_channels):
        super(MCIM1, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.a_conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.r_conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.r_conv_2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.r_conv_3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.r_conv_4 = nn.Conv3d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7)

        self.softmax = nn.Sigmoid()

        self.conv_2 = nn.Conv3d(in_channels * 4, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(in_channels)

    def forward(self, x):

        x2 = self.conv_1(x)

        x3_1 = self.a_conv_1(x2)
        x3_1 = self.r_conv_1(x3_1)
        x3_1 = self.softmax(x3_1)
        x3_1 = x3_1 * x2

        x3_2 = self.a_conv_1(x2)
        x3_2 = self.r_conv_2(x3_2)
        x3_2 = self.softmax(x3_2)
        x3_2 = x3_2 * x2

        x3_3 = self.a_conv_1(x2)
        x3_3 = self.r_conv_3(x3_3)
        x3_3 = self.softmax(x3_3)
        x3_3 = x3_3 * x2

        x3_4 = self.a_conv_1(x2)
        x3_4 = self.r_conv_4(x3_4)
        x3_4 = self.softmax(x3_4)
        x3_4 = x3_4 * x2

        y1 = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)
        y = self.conv_2(y1)


        return y
