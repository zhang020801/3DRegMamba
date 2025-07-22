import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from Model.Core_Mamba import FeatMambaBlock, MatchMambaBlock, MatchMambaBlock_Last

image_size_H=160
image_size_W=192
image_size_D=160


class RegHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out
class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x
class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class Conv3DReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        # self.norm2 = nn.InstanceNorm3d(out_channels)
        # self.act2 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x_out = self.act1(x)
        # x = self.conv2(x)
        # x = self.norm2(x)
        # x_out = self.act2(x)
        return x_out
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out
class Encoder(nn.Module):
    def __init__(self, in_channels=1, channel_num=8):
        super().__init__()
        self.conv_1 = Conv3DReLUBlock(in_channels=in_channels, out_channels=channel_num)
        # self.conv_1 = FeatMambaBlock(dim=1,  in_channels=in_channels, out_channels=channel_num, H=160, W=192, D=160)
        self.conv_2 = FeatMambaBlock(dim=16, in_channels=channel_num, out_channels=channel_num * 2, H=image_size_H//2, W=image_size_W//2, D=image_size_D//2)
        

        self.conv_3 = FeatMambaBlock(dim=32, in_channels=channel_num * 2, out_channels=channel_num * 4, H=image_size_H//4, W=image_size_W//4, D=image_size_D//4)
        

        self.conv_4 = FeatMambaBlock(dim=64, in_channels=channel_num * 4, out_channels=channel_num * 8, H=image_size_H//8, W=image_size_W//8, D=image_size_D//8)
        # self.conv_4 = nn.Conv3d(channel_num * 4, channel_num * 8, kernel_size=3, stride=1, padding=1)
        

        self.conv_5 = FeatMambaBlock(dim=128, in_channels=channel_num * 8, out_channels=channel_num * 16, H=image_size_H//16, W=image_size_W//16, D=image_size_D//16)
        

        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.conv_1(x_in)
        x = self.downsample(x_1)

        x_2 = self.conv_2(x)
        x = self.downsample(x_2)

        x_3 = self.conv_3(x)

        x = self.downsample(x_3)

        x_4 = self.conv_4(x)

        x = self.downsample(x_4)

        x_5 = self.conv_5(x)
        return [x_1, x_2, x_3, x_4, x_5]

class RegMamba(nn.Module):
    def __init__(self, channel_num=16):
        super().__init__()


        post = torch.tensor([5.0, 1.0, 1.0, 5.0, 1.0, 1.0])  
        # post = torch.tensor([])  
        self.post = torch.nn.Parameter(post, requires_grad=True)  

        self.encoder = Encoder(channel_num=16)

        self.conv_1 = Conv3DReLUBlock(channel_num * 1 * 2, channel_num * 1)
        self.conv_2 = Conv3DReLUBlock(channel_num * 2 * 3, channel_num * 2)
        self.conv_3 = Conv3DReLUBlock(channel_num * 4 * 3, channel_num * 4)
        self.conv_4 = Conv3DReLUBlock(channel_num * 8 * 3, channel_num * 8)
        self.conv_5 = Conv3DReLUBlock(channel_num * 16 * 2 + 1200, channel_num * 16)

        self.corr_2 = MatchMambaBlock(dim=channel_num * 2, H=image_size_H//2, W=image_size_W//2, D=image_size_D//2)
        self.corr_3 = MatchMambaBlock(dim=channel_num * 4, H=image_size_H//4, W=image_size_W//4, D=image_size_D//4)
        self.corr_4 = MatchMambaBlock(dim=channel_num * 8, H=image_size_H//8, W=image_size_W//8, D=image_size_D//8)
        self.corr_5 = MatchMambaBlock_Last(dim=channel_num * 16, H=image_size_H//16, W=image_size_W//16, D=image_size_D//16)

        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 2)
        self.reghead_3 = RegHead(channel_num * 4)
        self.reghead_4 = RegHead(channel_num * 8)
        self.reghead_5 = RegHead(channel_num * 16)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)


        # Step 1
        corr_5, corr_5_list = self.corr_5(x_mov_5, x_fix_5, alpha=self.post)
        cat = torch.cat([x_mov_5, corr_5, x_fix_5], dim=1)
        conv_corr_5 = self.conv_5(cat)
        flow_5 = self.reghead_5(conv_corr_5)

        # Step 2
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)


        corr_4, corr_4_list = self.corr_4(x_mov_4, x_fix_4, alpha=self.post)
        cat = torch.cat([x_mov_4, corr_4, x_fix_4], dim=1)
        conv_corr_4 = self.conv_4(cat)
        flow_4 = self.reghead_4(conv_corr_4)
        flow_4 = flow_4 + flow_5_up

        # Step 3
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)



        corr_3, corr_3_list = self.corr_3(x_mov_3, x_fix_3, alpha=self.post)
        cat = torch.cat([x_mov_3, corr_3, x_fix_3], dim=1)
        conv_corr_3 = self.conv_3(cat)
        flow_3 = self.reghead_3(conv_corr_3)
        flow_3 = flow_3 + flow_4_up


        # Step 4
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        flow_3_up =  self.ResizeTransformer(flow_4_up)

        corr_2, corr_2_list = self.corr_2(x_mov_2, x_fix_2, alpha=self.post)
        cat = torch.cat([x_mov_2, corr_2, x_fix_2], dim=1)
        conv_corr_2 = self.conv_2(cat)
        flow_2 = self.reghead_2(conv_corr_2)
        flow_2 = flow_2 + flow_3_up


        # Step 5
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)


        cat = torch.cat([x_mov_1, x_fix_1], dim=1)
        conv_corr_1 = self.conv_1(cat)
        flow_1 = self.reghead_1(conv_corr_1)
        flow_1 = flow_1 + flow_2_up

        # moved = self.SpatialTransformer(moving, flow_1)

        # return moved, flow_1
        return flow_1

