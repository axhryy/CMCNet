import torch
import torch.nn as nn
from models import common
from IPython import embed
import numpy as np
import math
import torch.nn.functional as F
import numbers
from einops import rearrange
from models.DW_ss2d import SS2D

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class CMCNet(nn.Module):
    def __init__(self, conv=common.default_conv, norm_layer=nn.LayerNorm):
        super(CMCNet, self).__init__()

        self.scale = 8
        self.phase = 3
        kernel_size = 3
        att_name = 'spar'
        self.img_range=1.
        self.head = conv(3, 32, kernel_size)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        ############ residual layers ############
        self.res_layers_down1_pre = ResidualBlock1_en_pre(32, 32, hg_depth=4, att_name=att_name, n_feat=32)
        self.res_layers_down1=ResidualBlock1_en(64, 64, hg_depth=4, att_name=att_name, n_feat=64)
        self.res_layers_down2=ResidualBlock2_en(128, 128, hg_depth=4, att_name=att_name, n_feat=128)
        self.res_layers=ResidualBlock2_middle(128, 128, hg_depth=2, att_name=att_name, n_feat=128)   
        self.res_layers_up1=ResidualBlock1_de(128, 128, hg_depth=4, att_name=att_name, n_feat=128)
        self.res_layers_up2=ResidualBlock2_de(64, 64, hg_depth=4, att_name=att_name, n_feat=64)
        self.res_layers_back=ResidualBlock_res_back(32, 32, hg_depth=4, att_name=att_name, n_feat=32)

        self.tail = conv(32, 3, kernel_size)
        self.tail352_128 = conv(352, 128, kernel_size)
        self.tail288_64 = conv(288, 64, kernel_size)
        self.tai256_32 = conv(256, 32, kernel_size)
        self.conv_64 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv_32 = nn.Conv2d(32, 32, 3, 2, 1)
        self.tranpose_128 = nn.ConvTranspose2d(128, 128, 8, 2, 3)

        self.tranpose64_128 = nn.ConvTranspose2d(64, 64, 8, 2, 3)
        self.tranpose128_128 = nn.ConvTranspose2d(128, 128, 6, 2, 2)

        self.down_input = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.LeakyReLU(0.2), )

        self.up_input = nn.Sequential(nn.ConvTranspose2d(32, 64, kernel_size=6, stride=2, padding=2, bias=False),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.LeakyReLU(0.2), )

        self.down1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.down3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.scale1_1 = Scale(0.25)
        self.scale1_2 = Scale(0.25)
        self.scale1_3 = Scale(0.25)
        self.scale1_4 = Scale(0.25)

        self.scale2_1 = Scale(0.25)
        self.scale2_2 = Scale(0.25)
        self.scale2_3 = Scale(0.25)
        self.scale2_4 = Scale(0.25)

        self.scale3_1 = Scale(0.25)
        self.scale3_2 = Scale(0.25)
        self.scale3_3 = Scale(0.25)
        self.scale3_4 = Scale(0.25)
        self.embed = nn.Conv2d(32, 32, 3, 1, 1)

        self.channel_func_128 = CALayer_128(16)
        self.channel_func_64 = CALayer_64(16)
        self.channel_func_32 = CALayer_32(16)
        self.spa_128=SpatialAttention()
        self.spa_64=SpatialAttention()
        self.spa_32=SpatialAttention()

    def forward(self, x):  

        input = x

        x = self.head(x)  

        x = self.res_layers_down1_pre(x)

        x1 = self.down1(x)  
        x1 = self.res_layers_down1(x1)  

        x2 = self.down2(x1)  
        x2 = self.res_layers_down2(x2)  

        x3 = self.down3(x2)  


        z = self.res_layers(x3) 

        y1 = self.up1(z) 
        y1_1 = self.conv_64(x1)  
        y1_2 = self.conv_32(x)  
        y1_2 = self.conv_32(y1_2)  
        
        #MFFM
        y1 = torch.cat([y1, x2, y1_1, y1_2], 1)  
        y1 = self.tail352_128(y1)  
        y_in_32=y1
        y_ca = self.channel_func_128(y_in_32)
        y_spa =self.spa_32(y_in_32)
        y1=y_ca+y_spa
        y1 = self.res_layers_up1(y1) 

        y2 = self.up2(y1)  
        y2_1 = self.conv_32(x)  
        y2_2 = self.tranpose_128(x2)  

        #MFFM
        y2 = torch.cat([y2, x1, y2_1, y2_2], 1)  
        y2 = self.tail288_64(y2)  
        
        y_in_64=y2
        y_ca_64 = self.channel_func_64(y_in_64)
        y_spa_64 =self.spa_64(y_in_64)
        y2=y_ca_64+y_spa_64

        y2 = self.res_layers_up2(y2)  

        ######## Third #####
        y3 = self.up3(y2)  

        y3_1 = self.tranpose64_128(x1) 

        y3_2 = self.tranpose128_128(x2) 
        y3_2 = self.tranpose128_128(y3_2)

        #MFFM
        y3 = torch.cat([y3, x, y3_1, y3_2], 1)  
        y3 = self.tai256_32(y3)  
        y_in_128=y3
        y_ca_128 = self.channel_func_32(y_in_128)
        y_spa_128 =self.spa_128(y_in_128)
        y3=y_ca_128+y_spa_128

        y3 = self.res_layers_back(y3)

        out = self.tail(y3)
        out = out + input

        return out

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ReluLayer(nn.Module):

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1 == 0, 'Relu type {} not support.'.format(relu_type)
    def forward(self, x):
        return self.func(x)


class NormLayer(nn.Module):

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class CALayer_128(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_128, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(128, 128 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // reduction, 128, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class CALayer_64(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_64, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // reduction, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class CALayer_32(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_32, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(32, 32 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // reduction, 32, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class HourGlassBlock(nn.Module):

    def __init__(self,n_feat, depth, c_in, c_out, c_mid=64,norm_type='bn',relu_type='prelu',):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
                    
        self.channel_func = CALayer(n_feat,16)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x  # (32,64,64,64)
        
        x = self._forward(self.depth, x)  # (32,64,64,64)
        
        x = self.channel_func(x)  # (32,64,64,64)
        
        x = x + input_x   # (32,64,64,64)
        self.att_map = self.out_block(x)  # (32,1,64,64)
        x = input_x * self.att_map  # (32,64,64,64)
        return x

class HourGlassBlock2(nn.Module):
    def __init__(self,n_feat, depth, c_in, c_out, c_mid=64,norm_type='bn',relu_type='prelu',):
        super(HourGlassBlock2, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
                    
        self.channel_func = CALayer2(n_feat,16)
        self._64 = nn.Conv2d(128,64,1)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):  # (32,128,32,32)
        if self.depth == 0: return x
        
        input_x = x  # (32,128,32,32)
        
        plus = self._64(x)  # (32,64,32,32)
        x = self._forward(self.depth, x)  # (32,64,32,32)
        
        x = self.channel_func(x)  # (32,64,32,32)
        
        x = x + plus   # (32,64,64,64)
        self.att_map = self.out_block(x)  # (32,1,64,64)
        
        x = input_x * self.att_map  # (32,64,64,64)
        return x
     
class HourGlassBlock32(nn.Module):

    def __init__(self,n_feat, depth, c_in, c_out, c_mid=64,norm_type='bn',relu_type='prelu',):
        super(HourGlassBlock32, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
                    
        self.channel_func = CALayer2(n_feat,16)
        self._64 = nn.Conv2d(32,64,1)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):  # (32,128,32,32)
        if self.depth == 0: return x
        
        input_x = x  # (32,128,32,32)
        
        plus = self._64(x)  # (32,64,32,32)
        x = self._forward(self.depth, x)  # (32,64,32,32)
        
        x = self.channel_func(x)  # (32,64,32,32)
        
        x = x + plus   # (32,64,64,64)
        self.att_map = self.out_block(x)  # (32,1,64,64)
        
        x = input_x * self.att_map  # (32,64,64,64)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none',
                 use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad

        bias = True if norm_type in ['pixel', 'none'] else False
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class new_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(new_CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class ResidualBlock1_en_pre(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_en_pre, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func= nn.Sequential(
            *[ HourGlassBlock32(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=32, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion=DCFM(c_in,1,ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias')

    def forward(self, x):  # (32,64,64,64)

        origin_x = x # (32,64,64,64)
        local_x = self.att_func(x)  # (32,64,64,64)
        global_x=self.encoder_level1(x)

        x = self.fusion(origin_x,global_x,local_x)
        return x


class ResidualBlock1_en(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_en, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func = nn.Sequential(
            *[   HourGlassBlock(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=64, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

    def forward(self, x):  # (32,64,64,64)

        origin_x = x  # (32,64,64,64)
        local_x = self.att_func(x)  # (32,64,64,64)
        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)
        return x

class ResidualBlock2_en(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock2_en, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func2 = nn.Sequential(
            *[HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=128, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

    def forward(self, x):  # (32,128,32,32)
        origin_x=x
        local_x = self.att_func2(x)  # (32,64,64,64)
        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)
  
        return x
    

class ResidualBlock2_middle(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock2_middle, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func2 = nn.Sequential(
            *[ HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=128, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

    def forward(self, x):  # (32,128,32,32)
        origin_x=x
        local_x = self.att_func2(x)  # (32,64,64,64)

        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)
  
        return x


class ResidualBlock1_de(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_de, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func2 = nn.Sequential(
            *[  HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=128, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

    def forward(self, x):  # (32,128,32,32)
        origin_x=x
        local_x = self.att_func2(x)  # (32,64,64,64)
        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)
        return x


class ResidualBlock2_de(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock2_de, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func = nn.Sequential(
            *[  HourGlassBlock(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=64, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

    def forward(self, x):  # (32,128,32,32)
        origin_x=x
        local_x = self.att_func(x)  # (32,64,64,64)
        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)

        return x

class ResidualBlock_res_back(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock_res_back, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        self.att_func2 = nn.Sequential(
            *[   HourGlassBlock32(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
              for i in range(1)])

        self.encoder_level1 = nn.Sequential(
            *[MultiScaleMambaBlock(dim=32, ffn_expansion_factor=2.66, bias=False)
              for i in range(2)])
        self.fusion = DCFM(c_in, 1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
    def forward(self, x):  # (32,128,32,32)
        origin_x=x
        local_x = self.att_func2(x)  # (32,64,64,64)
        global_x = self.encoder_level1(x)
        x = self.fusion(origin_x,global_x, local_x)
        return x

class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // reduction, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  

        y = self.avg_pool(x)  
        y = self.conv_du(y) 
        return x * y  


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x) 
        y = self.conv_du(y)  
        return x * y


class MultiScaleMambaBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MultiScaleMambaBlock, self).__init__()
        self.MSSM=SS2D(dim)
        self.layer_n=nn.LayerNorm(dim,eps=1e-6)
        self.layer_2=nn.LayerNorm(dim,eps=1e-6)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        input_x=x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.MSSM(self.layer_n(x))
        x = x.permute(0,3,1,2).contiguous()
        x = input_x+x 
        x = x + self.ffn(self.layer_2(x.permute(0, 2, 3, 1).contiguous()).permute(0,3,1,2).contiguous())
        return x 


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):

        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class fusion_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(fusion_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def _forward(self, q, kv):
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        out = (attn @ v)
        return out

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]
        q = self.q_dwconv(self.q(high))
        kv = self.kv_dwconv(self.kv(low))
        out = self._forward(q, kv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv.shape[-2], w=kv.shape[-1])
        out = self.project_out(out)
        return out

class DCFM(nn.Module):
    r""" Hybrid Fusion Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (int): Define the window size.
        bias (int): Shift size for SW-MSA.
        LayerNorm_type (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DCFM, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_1 = LayerNorm(dim, LayerNorm_type)
        self.attn = fusion_Attention(dim, num_heads, bias)
        self.attn2 = fusion_Attention(dim, num_heads, bias)
        self.conv1=nn.Conv2d(dim*2,dim,1,bias=False)
        self.CA=CALayer(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.dim = dim

    def forward(self,origin_x, low, high):
        self.h, self.w = low.shape[2:]
        x_low = low + self.attn(self.norm1(low), high)
        x_high= high + self.attn2(self.norm_1(high), low)
        x_out=torch.cat([x_low,x_high],dim=1)
        x_out=self.CA(self.conv1(x_out))+origin_x
        x_out = x_out + self.ffn(self.norm2(x_out))
        x_out = torch.nan_to_num(x_out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return x_out