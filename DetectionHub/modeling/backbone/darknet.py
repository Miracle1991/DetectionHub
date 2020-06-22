"""
Copyright (c) Baidu, Inc. and its affiliates. All Rights Reserved.
"""
import torch
import torch.nn.functional as F
from torch import nn

class DarkNet53(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(DarkNet53, self).__init__()
        self.conv1_1 = BaseBlockRelu(3, 32, 3, stride=1, padding=1, bias=False)
        # 2x
        self.conv1_2 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.res1a_b2a = BaseBlockRelu(64, 32, 1, stride=1, padding=0, bias=False)
        self.res1a_b2b = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        # 4x
        self.conv2_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.res2a_b2a = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.res2a_b2b = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.res2b_b2a = BaseBlockRelu(128, 64, 1, stride=1, padding=0, bias=False)
        self.res2b_b2b = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        # 8x
        self.conv3_1 = BaseBlockRelu(128, 256, 3, stride=2, padding=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3a_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3b_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3b_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3c_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3c_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3d_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3d_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3e_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3e_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3f_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3f_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3g_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3g_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.res3h_b2a = BaseBlockRelu(256, 128, 1, stride=1, padding=0, bias=False)
        self.res3h_b2b = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)

        # 16x
        self.conv4_1 = BaseBlockRelu(256, 512, 3, stride=2, padding=1, bias=False)
        self.res4a_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4a_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4b_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4b_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4c_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4c_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4d_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4d_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4e_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4e_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4f_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4f_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4g_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4g_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.res4h_b2a = BaseBlockRelu(512, 256, 1, stride=1, padding=0, bias=False)
        self.res4h_b2b = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)

        # 4x
        self.conv5_1 = BaseBlockRelu(512, 1024, 3, stride=2, padding=1, bias=False)
        self.res5a_b2a = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.res5a_b2b = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.res5b_b2a = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.res5b_b2b = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.res5c_b2a = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.res5c_b2b = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)
        self.res5d_b2a = BaseBlockRelu(1024, 512, 1, stride=1, padding=0, bias=False)
        self.res5d_b2b = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1_1(x)

        # 2x
        x = self.conv1_2(x)
        x = x + self.res1a_b2b(self.res1a_b2a(x))

        # 4x
        x = self.conv2_1(x)

        x = x + self.res2a_b2b(self.res2a_b2a(x))
        x = x + self.res2b_b2b(self.res2b_b2a(x))

        # 8x
        x = self.conv3_1(x)

        x = x + self.res3a_b2b(self.res3a_b2a(x))
        x = x + self.res3b_b2b(self.res3b_b2a(x))
        x = x + self.res3c_b2b(self.res3c_b2a(x))
        x = x + self.res3d_b2b(self.res3d_b2a(x))
        x = x + self.res3e_b2b(self.res3e_b2a(x))
        x = x + self.res3f_b2b(self.res3f_b2a(x))
        x = x + self.res3g_b2b(self.res3g_b2a(x))
        x = x + self.res3h_b2b(self.res3h_b2a(x))
        x_output3 = x

        # 16x
        x = self.conv4_1(x_output3)

        x = x + self.res4a_b2b(self.res4a_b2a(x))
        x = x + self.res4b_b2b(self.res4b_b2a(x))
        x = x + self.res4c_b2b(self.res4c_b2a(x))
        x = x + self.res4d_b2b(self.res4d_b2a(x))
        x = x + self.res4e_b2b(self.res4e_b2a(x))
        x = x + self.res4f_b2b(self.res4f_b2a(x))
        x = x + self.res4g_b2b(self.res4g_b2a(x))
        x = x + self.res4f_b2b(self.res4f_b2a(x))
        x_output2 = x

        # 32x
        x = self.conv5_1(x_output2)

        x = x + self.res5a_b2b(self.res5a_b2a(x))
        x = x + self.res5b_b2b(self.res5b_b2a(x))
        x = x + self.res5c_b2b(self.res5c_b2a(x))
        x = x + self.res5d_b2b(self.res5d_b2a(x))
        x_output1 = x

        return [x_output3, x_output2, x_output1]


class DarkNet_Tiny(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(DarkNet_Tiny, self).__init__()
        # 0
        self.conv1 = BaseBlockRelu(3, 16, 3, stride=1, padding=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # 2
        self.conv2 = BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 4
        self.conv3 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        # 6
        self.conv4 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        # 8
        self.conv5 = BaseBlockRelu(128, 256, 3, stride=1, padding=1, bias=False)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        # 10
        self.conv6 = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.maxpool6 = nn.MaxPool2d(2, 2)
        # 12
        self.conv7 = BaseBlockRelu(512, 1024, 3, stride=1, padding=1, bias=False)

        ######
        self.conv8 = BaseBlockRelu(1024, 256, 3, stride=1, padding=1, bias=False)
        self.conv9 = BaseBlockRelu(256, 512, 3, stride=1, padding=1, bias=False)
        self.conv10 = BaseBlockRelu(512, 255, 3, stride=1, padding=1, bias=False)



        self.conv1_4 = BaseBlock(32, 64, 3, stride=1, padding=1, bias=False)
        self.res1a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res1a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

























class BaseBlock(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        return x


class BaseBlockDeconvRelu(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockDeconvRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class BaseBlockRelu(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class BaseBlockRelu1(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockRelu1, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu_(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x



class BaseBlockReluPool(nn.Module):
    """
    As titled
    """

    def __init__(self, *args, **kargs):
        super(BaseBlockReluPool, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        x = self.pool(x)
        return x
