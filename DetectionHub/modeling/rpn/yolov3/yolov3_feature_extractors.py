# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F


class Yolov3Head(nn.Module):
    def __init__(self, config):
        super(Yolov3Head, self).__init__()
        self.cfg = config
        numclass = (80 + 5) * 3
        self.min_conv1 = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.min_conv2 = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.min_conv3 = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.min_conv4 = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.min_conv5 = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)

        self.min_conv6 = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.min_conv_out = ConvBlock(1024, numclass, 1, stride=1, padding=0, bias=False)

        self.min_2_mid = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.min_upsamle = nn.Upsample(scale_factor=2, mode='nearest')
        # cat should be here

        self.mid_conv1 = BaseBlockRelu(768, 256, 1, stride=1, padding=0, bias=False)

        self.mid_conv2 = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.mid_conv3 = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.mid_conv4 = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.mid_conv5 = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)

        self.mid_conv6 = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.mid_conv_out = ConvBlock(512, numclass, 1, stride=1, padding=0, bias=False)

        self.mid_2_large = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.mid_upsamle = nn.Upsample(scale_factor=2, mode='nearest')
        # cat should be here

        self.large_conv1 = BaseBlockRelu(384, 128, 1, stride=1, padding=0, bias=False)

        self.large_conv2 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.large_conv3 = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.large_conv4 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.large_conv5 = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)

        self.large_conv6 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.large_conv_out = ConvBlock(256, numclass, 1, stride=1, padding=0, bias=False)


    def forward(self, x_in):
        # min
        scale_min_feature = x_in[2]
        x = self.min_conv1(scale_min_feature)
        x = self.min_conv2(x)
        x = self.min_conv3(x)
        x = self.min_conv4(x)
        x_min = self.min_conv5(x)
        x = self.min_conv6(x_min)
        x_min_out = self.min_conv_out(x)

        x = self.min_2_mid(x_min)
        x = self.min_upsamle(x)
        scale_mid_feature = x_in[1]
        x = torch.cat((x, scale_mid_feature), 1)

        # mid
        x = self.mid_conv1(x)
        x = self.mid_conv2(x)
        x = self.mid_conv3(x)
        x = self.mid_conv4(x)
        x_mid = self.mid_conv5(x)
        x = self.mid_conv6(x_mid)
        x_mid_out = self.mid_conv_out(x)

        x = self.mid_2_large(x_mid)
        x = self.mid_upsamle(x)
        scale_large_feature = x_in[0]
        x = torch.cat((x, scale_large_feature), 1)

        #large
        x = self.large_conv1(x)
        x = self.large_conv2(x)
        x = self.large_conv3(x)
        x = self.large_conv4(x)
        x_large = self.large_conv5(x)
        x = self.large_conv6(x_large)
        x_large_out = self.large_conv_out(x)

        return [x_large_out, x_mid_out, x_min_out]


class BaseBlock(nn.Module):
    def __init__(self, *args, **kargs):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BaseBlockRelu(nn.Module):
    def __init__(self, *args, **kargs):
        super(BaseBlockRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, *args, **kargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class BaseBlockDeconvRelu(nn.Module):
    def __init__(self, *args, **kargs):
        super(BaseBlockDeconvRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


def build_yolov3_feature_extractors(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return Yolov3Head(cfg)