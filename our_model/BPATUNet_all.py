import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
import numpy as np
import argparse
import math
from our_model.unet_parts import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from our_model.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from SoftPool import soft_pool2d, SoftPool2d
from our_model.deform_attention_2 import DeformableAttention2D
from our_model.deform_conv import DeformConv2d
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)
class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out
class Aspp_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        self.pool_1 = nn.AdaptiveAvgPool2d(8)
        self.pool_2 = nn.AdaptiveAvgPool2d(4)
        self.pool_3 = nn.AdaptiveAvgPool2d(2)
        self.pool_4 = nn.AdaptiveAvgPool2d(1)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.dsc = depthwise_separable_conv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        #进行多尺度池化
        x_1 = self.pool_1(x_)#(8,512,8,8)
        x_1 = self.dsc(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c')
        x_2 = self.pool_2(x_)#(8,512,4,4)
        x_2 = self.dsc(x_2)
        x_2 = rearrange(x_2, 'b c h w -> b (h w) c')
        x_3 = self.pool_3(x_)#(8,512,2,2)
        x_3 = self.dsc(x_3)
        x_3 = rearrange(x_3, 'b c h w -> b (h w) c')
        x_4 = self.pool_4(x_)#(8,512,1,1)
        x_4 = self.dsc(x_4)
        x_4 = rearrange(x_4, 'b c h w -> b (h w) c')
        x = torch.cat((x_1, x_2, x_3, x_4), dim = 1)#(8,85,512)
        # x = self.sr(x)  # (8,512,8,8)
        # x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)#(8,85,1024)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)#(2,8,8,85,64)
        k, v = x.unbind(0)#(8,8,85,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
class Aspp_Attention_16(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        self.pool_0 = nn.AdaptiveAvgPool2d(16)
        self.pool_1 = nn.AdaptiveAvgPool2d(8)
        self.pool_2 = nn.AdaptiveAvgPool2d(4)
        self.pool_3 = nn.AdaptiveAvgPool2d(2)
        self.pool_4 = nn.AdaptiveAvgPool2d(1)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.dsc = depthwise_separable_conv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        #进行多尺度池化
        x_0 = self.pool_0(x_)  # (8,512,8,8)
        x_0 = self.dsc(x_0)
        x_0 = rearrange(x_0, 'b c h w -> b (h w) c')
        x_1 = self.pool_1(x_)#(8,512,8,8)
        x_1 = self.dsc(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c')
        x_2 = self.pool_2(x_)#(8,512,4,4)
        x_2 = self.dsc(x_2)
        x_2 = rearrange(x_2, 'b c h w -> b (h w) c')
        x_3 = self.pool_3(x_)#(8,512,2,2)
        x_3 = self.dsc(x_3)
        x_3 = rearrange(x_3, 'b c h w -> b (h w) c')
        x_4 = self.pool_4(x_)#(8,512,1,1)
        x_4 = self.dsc(x_4)
        x_4 = rearrange(x_4, 'b c h w -> b (h w) c')
        x = torch.cat((x_0, x_1, x_2, x_3, x_4), dim=1)#(8,85,512)
        # x = self.sr(x)  # (8,512,8,8)
        # x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)#(8,85,1024)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)#(2,8,8,85,64)
        k, v = x.unbind(0)#(8,8,85,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
class tiao_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        # self.pool_1 = nn.AdaptiveAvgPool2d(8)
        # self.pool_2 = nn.AdaptiveAvgPool2d(4)
        # self.pool_3 = nn.AdaptiveAvgPool2d(2)
        # self.pool_4 = nn.AdaptiveAvgPool2d(1)
        self.pool_shu = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_heng = nn.AdaptiveAvgPool2d((1, None))

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        #进行多尺度池化
        x_shu = self.pool_shu(x_)#(8,512,16,1)
        x_shu = self.sr(x_shu)#(8,512,16,1)
        x_shu = rearrange(x_shu, 'b c h w -> b (h w) c')
        x_heng = self.pool_heng(x_)#(8,512,1,16)
        x_heng = self.sr(x_heng)
        x_heng = rearrange(x_heng, 'b c h w -> b (h w) c')
        # x_1 = self.pool_1(x_)#(8,512,8,8)
        # x_1 = self.sr(x_1)
        # x_1 = x_1.view(b, 64, c)
        # x_2 = self.pool_2(x_)#(8,512,4,4)
        # x_2 = self.sr(x_2)
        # x_2 = x_2.view(b, 16, c)
        # x_3 = self.pool_3(x_)#(8,512,2,2)
        # x_3 = self.sr(x_3)
        # x_3 = x_3.view(b, 4, c)
        # x_4 = self.pool_4(x_)#(8,512,1,1)
        # x_4 = self.sr(x_4)
        # x_4 = x_4.view(b, 1, c)
        x = torch.cat((x_shu, x_heng), dim=1)#(8,85,512)
        # x = self.sr(x)  # (8,512,8,8)
        # x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)#(8,8,85,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
class fang_chihua_module(nn.Module):
    def __init__(self, dim):
        super(fang_chihua_module, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.fang_Attention = Aspp_Attention(dim)
    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x_fang = self.fang_Attention(x)
        return x_fang
class fang_chihua_module_16(nn.Module):
    def __init__(self, dim):
        super(fang_chihua_module_16, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.fang_Attention = Aspp_Attention_16(dim)
    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x_fang = self.fang_Attention(x)
        return x_fang
class tiao_chihua_module(nn.Module):
    def __init__(self, dim):
        super(tiao_chihua_module, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.tiao_Attention = tiao_Attention(dim)


    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x_tiao = self.tiao_Attention(x)

        return x_tiao
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max','soft']):
        super(ChannelAtt, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU()
            #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        channel_att_sum = None
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        self.soft_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avgpoolmlp = self.mlp(avg_pool)
        maxpoolmlp=self.mlp(max_pool)
        softpoolmlp = self.mlp(self.soft_pool(x.contiguous()))
        pooladd = avgpoolmlp+maxpoolmlp+softpoolmlp

        #self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # self.pool = AdaPool2d(kernel_size=(x.size(2), x.size(3)), beta=(1, 1), stride=(x.size(2), x.size(3)),
        #                       device='cuda')
        # ada_pool = self.mlp(self.pool(x))
        # weightPool = ada_pool * pooladd
        weightPool = pooladd
        # channel_att_sum = self.mlp(weightPool)
        channel_att_sum = self.incr(weightPool)
        Att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return Att
class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=1)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 2):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.pe = PositionalEncodingPermute2D(dim)

        self.sr_ratio = sr_ratio
        #segformer中的和PVT v1中的mhsa
        # if sr_ratio > 1:
        #     sr_ratio_tuple = (sr_ratio, sr_ratio)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
        #     self.norm = nn.LayerNorm(dim)

        #PVT v2中的mhsa
        self.soft_pool = SoftPool2d(kernel_size=(8,8), stride=(8, 8))
        #self.pool = nn.AdaptiveAvgPool2d(8)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe#(8,512,16,16),(b,c,h,w)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, d]
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        #PVT v1中的
        # if self.sr_ratio > 1:
        #     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)#(8,512,16,16)
        #     x = self.sr(x)#(8,512,8,8)
        #     x = rearrange(x, 'b c h w -> b (h w) c')#(8,64,512)
        #     x = self.norm(x)

        #PVT v2中的
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (8,512,16,16)
        avg_pool = F.avg_pool2d(x, (8, 8), stride=(8, 8))
        max_pool = F.max_pool2d(x, (8, 8), stride=(8, 8))
        soft_pool = self.soft_pool(x.contiguous())
        #x = self.pool(x)
        x = avg_pool + max_pool + soft_pool
        x = self.sr(x)  # (8,512,8,8)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,64,512)
        x = self.norm(x)
        x = self.act(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
class MEAttention(nn.Module):
    def __init__(self, dim):
        super(MEAttention, self).__init__()
        self.num_heads = 8
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):#(b,c,h,w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)#(24,256,512)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)

        return x
class Def_EAmodule(nn.Module):
    def __init__(self, dim=1024):
        super(Def_EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.dcn_bot = DeformConv2d(inc=512, outc=512, kernel_size=3, padding=0, stride=1, modulation=True)
        self.Def_Attention = DeformableAttention2D(dim=512)
        self.eff_att = EfficientAttention(512)
        self.E_Attention = MEAttention(dim)

    def forward(self, x):
        m = x  # (8,512,16,16), (B, N, H)
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')#(8,16*16,512)
        x = self.SlayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)#(8,512,16,16)

        x_gao = x[:, :512, :, :]
        x_di = x[:, 512:, :, :]
        x_win = x_gao.view(b, 512, 2, 8, 2, 8)
        x_win = x_win.permute(2, 4, 0, 1, 3, 5)  # (2,2,8,512,8,8)
        list = []
        for i in range(2):
            for j in range(2):
                z = self.Def_Attention(x_win[i][j])
                list.append(z)
        result_1 = list  # 四个tensor,(8,64,512)
        result = []
        for i in range(4):
            result.append(result_1[i])  # 4个tensor,(8,512,8,8)
        shang_liangfutu = torch.cat((result[0], result[1]), dim=3)  # (8,512,8,16)
        xia_liangfutu = torch.cat((result[2], result[3]), dim=3)  # (8,512,8,16)
        si_fu_tu = torch.cat((shang_liangfutu, xia_liangfutu), dim=2)  # (8,512,16,16)

        x_dcn = self.dcn_bot(x_gao)
        x_gao_he = x_dcn + si_fu_tu

        x_eff = self.eff_att(x_di)

        # x = self.Def_Attention(x)
        # x = m + x
        x = torch.cat((x_gao_he, x_eff), dim=1)
        x = x + m

        m = x
        x = rearrange(x, 'b c h w -> b (h w) c')  # (8,16*16,512)
        x = self.ElayerNorm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)

        x = self.E_Attention(x)
        x = m + x

        return x
class BPATUNet(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 n_classes: int = 1,
                 bilinear: bool = False
                 ):
        super(BPATUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.fang_chihua_5 = fang_chihua_module(dim=1024)
        self.tiaochihua_5 = tiao_chihua_module(dim=1024)
        self.conv1_5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.up_bian_5 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        # self.a5_1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.a5_2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.fang_chihua_4 = fang_chihua_module_16(dim=512)
        self.tiaochihua_4 = tiao_chihua_module(dim=512)
        self.conv1_4 = nn.Conv2d(512, 1, kernel_size=1)
        self.up_bian_4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.fang_chihua_3 = fang_chihua_module_16(dim=256)
        self.tiaochihua_3 = tiao_chihua_module(dim=256)
        self.conv1_3 = nn.Conv2d(256, 1, kernel_size=1)
        self.up_bian_3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        # self.fang_chihua_2 = fang_chihua_module_16(dim=128)
        # self.tiaochihua_2 = tiao_chihua_module(dim=128)
        # self.conv1_2 = nn.Conv2d(128, 1, kernel_size=1)
        # self.up_bian_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # self.vit = ViT(img_dim=self.img_dim_vit,
        #                in_channels=vit_channels,  # input features' channels (encoder)
        #                patch_dim=patch_size,
        #                # transformer inside dimension that input features will be projected
        #                # out will be [batch, dim_out_vit_tokens, dim ]
        #                dim=vit_transformer_dim,
        #                blocks=vit_blocks,
        #                heads=vit_heads,
        #                dim_linear_block=vit_dim_linear_mhsa_block,
        #                classification=False) if vit_transformer is None else vit_transformer
        #
        # # to project patches back - undoes vit's patchification
        # token_dim = vit_channels * (patch_size ** 2)
        # self.project_patches_back = nn.Linear(vit_transformer_dim, token_dim)
        # # upsampling path
        # self.vit_conv = SingleConv(in_ch=vit_channels, out_ch=1024)

        self.def_ea = Def_EAmodule(1024)

        self.up1_ = Up_(128, 64)
        self.up2_ = Up_(256, 128)
        self.up3_ = Up_(512, 256)
        self.up4_ = Up_(1024, 512)

        self.def_att_5 = DeformConv2d(inc=1024, outc=1024, kernel_size=3, padding=0, stride=1, modulation=True)
        #self.def_att_5 = DeformableAttention2D(dim=1024)
        # bn=nn.BatchNorm2d(outchannel[i_layer])
        self.channel_att_4 = ChannelAtt(gate_channels=512, reduction_ratio=2, pool_types=['avg', 'max'])

        self.dcn_4 = DeformConv2d(inc=512, outc=512, kernel_size=3, padding=0, stride=1, modulation=True)
        #self.def_att_4 = DeformableAttention2D(dim=512)
        # bn=nn.BatchNorm2d(outchannel[i_layer])
        self.channel_att_3 = ChannelAtt(gate_channels=256, reduction_ratio=2, pool_types=['avg', 'max'])

        self.weight_level_256_4 = nn.Conv2d(256, 4, 1)
        self.weight_level_512_4 = nn.Conv2d(512, 4, 1)
        self.weight_level_128_4 = nn.Conv2d(128, 4, 1)
        self.weight_level_64_4 = nn.Conv2d(64, 4, 1)
        self.weight_level_8_2 = nn.Conv2d(8, 2, 1)

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2_fang_chihua = self.fang_chihua_2(x2)
        # x2_tiao_chihua = self.tiaochihua_2(x2)
        # x2_chihua = self.a5_1 * x2_fang_chihua + self.a5_2 * x2_tiao_chihua
        # x2_conv1 = self.conv1_2(x2_chihua)
        # x2_bian = self.up_bian_2(x2_conv1)  # 用来作监督的
        # x2_cheng = torch.mul(x2, x2_conv1)
        # x2_jia = x2_cheng + x2  # Boundary Enhanced Feature

        x2_1 = self.up1_(x2)
        weight_1_channel = self.weight_level_64_4(x1)
        weight_1_cheng = self.weight_level_64_4(x2_1)
        weight_1_jia = torch.cat((weight_1_channel, weight_1_cheng), 1)
        level_weight_1_jia = self.weight_level_8_2(weight_1_jia)
        level_weight_1_jia = F.softmax(level_weight_1_jia, dim=1)
        fused_x1_jia = x1 * level_weight_1_jia[:, 0:1, :, :] + \
                       x2_1 * level_weight_1_jia[:, 1:2, :, :]

        x3 = self.down2(x2)
        x3_fang_chihua = self.fang_chihua_3(x3)
        x3_tiao_chihua = self.tiaochihua_3(x3)
        #x3_chihua = self.a5_1 * x3_fang_chihua + self.a5_2 *
        x3_chihua = x3_fang_chihua
        x3_conv1 = self.conv1_3(x3_chihua)
        x3_bian = self.up_bian_3(x3_conv1)  # 用来作监督的
        x3_cheng = torch.mul(x3_tiao_chihua, x3_conv1)
        x3_jia = x3_cheng + x3  # Boundary Enhanced Feature

        x3_2 = self.up2_(x3_jia)
        weight_2_channel = self.weight_level_128_4(x2)
        weight_2_cheng = self.weight_level_128_4(x3_2)
        weight_2_jia = torch.cat((weight_2_channel, weight_2_cheng), 1)
        level_weight_2_jia = self.weight_level_8_2(weight_2_jia)
        level_weight_2_jia = F.softmax(level_weight_2_jia, dim=1)
        fused_x2_jia = x2 * level_weight_2_jia[:, 0:1, :, :] + \
                       x3_2 * level_weight_2_jia[:, 1:2, :, :]
        #
        x4 = self.down3(x3_jia)
        x4_fang_chihua = self.fang_chihua_4(x4)
        x4_tiao_chihua = self.tiaochihua_4(x4)
        #x4_chihua = self.a5_1 * x4_fang_chihua + self.a5_2 * x4_tiao_chihua
        x4_chihua = x4_fang_chihua
        x4_conv1 = self.conv1_4(x4_chihua)
        x4_bian = self.up_bian_4(x4_conv1)  # 用来作监督的
        x4_cheng = torch.mul(x4_tiao_chihua, x4_conv1)
        x4_jia = x4_cheng + x4  # Boundary Enhanced Feature

        # x4_3 = self.up3_(x4_jia)
        # weight_3_channel = self.weight_level_256_4(x3_jia)
        # weight_3_cheng = self.weight_level_256_4(x4_3)
        # weight_3_jia = torch.cat((weight_3_channel, weight_3_cheng), 1)
        # level_weight_3_jia = self.weight_level_8_2(weight_3_jia)
        # level_weight_3_jia = F.softmax(level_weight_3_jia, dim=1)
        # fused_x3_jia = x3_jia * level_weight_3_jia[:, 0:1, :, :] + \
        #                x4_3 * level_weight_3_jia[:, 1:2, :, :]

        x3_channel = self.channel_att_3(x3_jia)  # 1#(256,64,64)
        # x4_def = self.def_att_4(x4)
        x4_def = self.dcn_4(x4_jia)
        x4_def_3 = self.up3_(x4_def)
        x3_cheng = torch.mul(x3_channel, x4_def_3)  # 2
        weight_3_channel = self.weight_level_256_4(x3_channel)
        weight_3_cheng = self.weight_level_256_4(x3_cheng)
        weight_3_jia = torch.cat((weight_3_channel, weight_3_cheng), 1)
        level_weight_3_jia = self.weight_level_8_2(weight_3_jia)
        level_weight_3_jia = F.softmax(level_weight_3_jia, dim=1)
        fused_x3_jia = x3_channel * level_weight_3_jia[:, 0:1, :, :] + \
                       x3_cheng * level_weight_3_jia[:, 1:2, :, :] + x3_jia

        x5 = self.down4(x4_jia)
        #x5进行self-attention
        # x5_eff_att = self.def_att_5(x5)
        x5_fang_chihua = self.fang_chihua_5(x5)
        x5_tiao_chihua = self.tiaochihua_5(x5)
        #x5_chihua = self.a5_1 * x5_fang_chihua + self.a5_2 * x5_tiao_chihua
        x5_chihua = x5_fang_chihua
        x5_conv1 = self.conv1_5(x5_chihua)
        x5_bian = self.up_bian_5(x5_conv1)#用来作监督的
        x5_cheng = torch.mul(x5_tiao_chihua, x5_conv1)
        x5_jia = x5_cheng + x5#Boundary Enhanced Feature

        x4_channel = self.channel_att_4(x4_jia)  # 1#(512,32,32)
        x5_def = self.def_att_5(x5_jia)
        x5_def_4 = self.up4_(x5_def)
        x4_cheng = torch.mul(x4_channel, x5_def_4)  # 2
        weight_4_channel = self.weight_level_512_4(x4_channel)
        weight_4_cheng = self.weight_level_512_4(x4_cheng)
        weight_4_jia = torch.cat((weight_4_channel, weight_4_cheng), 1)
        level_weight_4_jia = self.weight_level_8_2(weight_4_jia)
        level_weight_4_jia = F.softmax(level_weight_4_jia, dim=1)
        fused_x4_jia = x4_channel * level_weight_4_jia[:, 0:1, :, :] + \
                       x4_cheng * level_weight_4_jia[:, 1:2, :, :] + x4_jia

        # y = self.vit(x5_jia)  # out shape of number_of_patches, vit_transformer_dim
        #
        # # from [number_of_patches, vit_transformer_dim] -> [number_of_patches, token_dim]
        # y = self.project_patches_back(y)
        #
        # # from [batch, number_of_patches, token_dim] -> [batch, channels, img_dim_vit, img_dim_vit]
        # y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
        #               x=self.img_dim_vit // self.patch_size, y=self.img_dim_vit // self.patch_size,
        #               patch_x=self.patch_size, patch_y=self.patch_size)
        #
        # y = self.vit_conv(y)

        x5_def_ea = self.def_ea(x5_jia)

        x = self.up1(x5_def_ea, fused_x4_jia)
        x = self.up2(x, fused_x3_jia)
        x = self.up3(x, fused_x2_jia)
        x = self.up4(x, fused_x1_jia)
        outputs = self.outc(x)
        bian = x5_bian + x4_bian + x3_bian
        return outputs, bian