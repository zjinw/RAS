import torch.nn as nn
import torch

from model.utils.Transformer.transformer import TransformerModel
from model.utils.Transformer.PositionalEncoding import LearnedPositionalEncoding

from model.utils.MFFM import MFFM,MPI
from model.utils.MCIM import MCIM,MCIM1

class RASnet(nn.Module):

    def __init__(self, in_channels, n_classes, base_n_filter=16):
        super(RASnet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=(2,2,1), mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

        self.position_encoding1 = LearnedPositionalEncoding(405, 512, 900)
        self.pe_dropout = nn.Dropout(p=0.2)
        self.pre_head_ln1 = nn.LayerNorm(512)

        self.transformer1 = TransformerModel(512,12,8,4096,0.1,0.1)

        self.deconv0 = nn.Conv3d(512, self.base_n_filter * 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm4 = nn.BatchNorm3d(256)
        self.cont = nn.Conv3d(self.base_n_filter*16,512,kernel_size=3,padding=1)

        self.maxpool = nn.MaxPool3d(2)
        self.mpi_1 = MPI(1,self.base_n_filter*2)
        self.mffm_1 = MFFM(self.base_n_filter*2)

        self.mpi_2 = MPI(1,self.base_n_filter * 4)
        self.mffm_2 = MFFM(self.base_n_filter * 4)

        self.mpi_3 = MPI(1,self.base_n_filter * 8)
        self.mffm_3 = MFFM(self.base_n_filter * 8)

        self.mpi_4 = MPI(1,self.base_n_filter * 16)
        self.mffm_4 = MFFM(self.base_n_filter * 16)

        self.MCIM_1 = MCIM(self.base_n_filter)
        self.MCIM_2 = MCIM(self.base_n_filter*2)
        self.MCIM_3 = MCIM(self.base_n_filter*4)
        self.MCIM_4 = MCIM1(self.base_n_filter*8)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        # print(x.shape)
        out = self.conv3d_c1_1(x)

        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        mutiscale_x1 = self.maxpool(x)
        mutiscale_x1_1 = self.mpi_1(mutiscale_x1)
        # Level 2 context pathway
        out = self.conv3d_c2(out)
        out1 = self.mffm_1(out, mutiscale_x1_1)
        out = out + out1

        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        mutiscale_x2 = self.maxpool(mutiscale_x1)
        mutiscale_x2_1 = self.mpi_2(mutiscale_x2)
        # Level 3 context pathway
        out = self.conv3d_c3(out)
        out2 = self.mffm_2(out, mutiscale_x2_1)
        out = out + out2

        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        mutiscale_x3 = self.maxpool(mutiscale_x2)
        mutiscale_x3_1 = self.mpi_3(mutiscale_x3)
        # Level 4 context pathway
        out = self.conv3d_c4(out)
        out3 = self.mffm_3(out, mutiscale_x3_1)
        out = out + out3

        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        mutiscale_x4 = self.maxpool(mutiscale_x3)
        mutiscale_x4_1 = self.mpi_4(mutiscale_x4)
        # Level 5
        out = self.conv3d_c5(out)
        # residual_5 = out
        out4 = self.mffm_4(out, mutiscale_x4_1)
        out = out + out4

        residual_6 = out

        out = self.norm4(out)
        out = self.relu1(out)
        out = self.cont(out)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.size(0), -1, 512)

        out = self.position_encoding1(out)
        out = self.pe_dropout(out)

        # apply transformer
        out, intmd_x0 = self.transformer1(out)
        out = self.pre_head_ln1(out)

        out = out.view(out.size(0), 15, 10, 6, 512)     #shape
        out = out.permute(0, 4, 1, 2, 3).contiguous()

        out = self.deconv0(out)

        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)

        out += residual_6
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        context_1 = self.MCIM_1(context_1,context_2)
        context_2 = self.MCIM_2(context_2,context_3)
        context_3 = self.MCIM_3(context_3,context_4)
        context_4 = self.MCIM_4(context_4)


        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out

        # return [seg_layer,out_pred,ds1_ds2_sum_upscale_ds3_sum_upscale]
        return seg_layer



