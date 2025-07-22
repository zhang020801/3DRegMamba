
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
import torch.nn.functional as nnf
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_, to_3tuple


from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights


import math

from collections import namedtuple

# from mamba_ssm.modules.mamba_simple import Mamba
from Mambablock.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# from rope import *
import random
import numpy
from einops import rearrange


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None




class FeatMambaBlock(nn.Module):
    def __init__(self, dim, in_channels, out_channels, H, W, D):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v1',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W, input_c=D)

    def forward(self, input):
        # input: (B, C, D, H, W)
        input_skip = self.conv(input)
        input_skip = self.norm(input_skip)
        input_skip = self.act(input_skip)
        
        b, c, h, w, d = input.shape
        input = rearrange(input, 'b c h w d -> b (h w d) c', d=d, h=h, w=w)
        output, outputs = self.block(input)
        output = rearrange(output, 'b (h w d) c -> b c h w d ', d=d, h=h, w=w)
        output = self.conv(output)
        return output + input_skip


class MatchMambaBlock_Last(nn.Module):
    def __init__(self, dim, H, W, D):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv3d(in_channels=256, out_channels=1200, kernel_size=1, stride=1, padding=0)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W, input_c=D)

    def forward(self, input0, input1, alpha=None):
        # input0: (B, C, H, W, D) | input1: (B, C, H, W, D)
        skip = input0
        b, c, h, w, d = input0.shape
        input0 = rearrange(input0, 'b c h w d -> b (h w d) c', d=d, h=h, w=w)
        input1 = rearrange(input1, 'b c h w d -> b (h w d) c', d=d, h=h, w=w)

        input0 = self.norm0(input0)
        input1 = self.norm1(input1)

        output, outputs = self.block(input0, extra_emb=input1, alpha=alpha)
        output = rearrange(output, 'b (h w d) c -> b c h w d ', d=d, h=h, w=w)
        output = self.conv(output + skip)

        outputs = [rearrange(output_item, 'b (h w d) c -> b c h w d ', d=d, h=h, w=w) for output_item in outputs]
        outputs = [self.conv(output_item + skip) for output_item in outputs]

        return output, outputs

class MatchMambaBlock(nn.Module):
    def __init__(self, dim, H, W, D):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W, input_c=D)
        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input0, input1, alpha=None):
        # input0: (B, C, D, H, W) | input1: (B, C, D, H, W)
        skip = input0
        b, c, h, w, d = input0.shape
        input0 = rearrange(input0, 'b c h w d -> b (h w d) c', d=d, h=h, w=w)
        input1 = rearrange(input1, 'b c h w d -> b (h w d) c', d=d, h=h, w=w)

        input0 = self.norm0(input0)
        input1 = self.norm1(input1)

        output, outputs = self.block(input0, extra_emb=input1, alpha=alpha)
        output = rearrange(output, 'b (h w d) c -> b c h w d ', d=d, h=h, w=w)
        output = self.conv(output + skip)

        outputs = [rearrange(output_item, 'b (h w d) c -> b c h w d ', d=d, h=h, w=w) for output_item in outputs]
        outputs = [self.conv(output_item + skip) for output_item in outputs]

        return output, outputs

