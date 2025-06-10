import torch
from torch import nn


class MPI(nn.Module):
    def __init__(self,in_channels,out):
        super(MPI, self).__init__()

        self.conv_1 = nn.Conv3d(in_channels = in_channels, out_channels = out, kernel_size = 1)
        self.conv_2 = nn.Conv3d(in_channels = in_channels, out_channels = out, kernel_size=3,padding=1)
        self.conv_3 = nn.Conv3d(in_channels=in_channels,out_channels=out,kernel_size=5,padding=2)
        self.norm = nn.BatchNorm3d(out)
        self.conv_4 = nn.Conv3d(in_channels=out*3,out_channels=out,kernel_size=1)

    def forward(self,x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)

        y1 = torch.cat((x1,x2,x3),dim=1)

        y1 = self.conv_4(y1)
        y1 = self.norm(y1)
        return y1




class MFFM(nn.Module):
    def __init__(self,out):
        super(MFFM, self).__init__()

        self.conv = nn.Conv3d(out,out,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,y):
        x1 = x + y


        x1 = self.conv(x1)
        x1 = self.relu(x1)

        x1 = self.conv(x1)
        y1 = self.sigmoid(x1)

        y2 = y1 * x

        return y2