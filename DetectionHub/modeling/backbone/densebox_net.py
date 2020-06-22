# Copyright (c) Baidu, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

class Densebox_Net_32T(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(Densebox_Net_32T, self).__init__()
        self.conv1 = BaseBlockRelu(3, 64, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockRelu(64, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_2 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)

        # self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        # self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        # self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4c_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4d_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4d_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = x + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = x
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = x  # self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        return [res4d]


class Densebox_Net_32T_wheat(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(Densebox_Net_32T_wheat, self).__init__()
        self.conv1 = BaseBlockRelu(3, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv2_2 = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockRelu(64, 32, 1, stride=1, padding=1, bias=False)

        self.res2a_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        # self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        # self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4c_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4d_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4d_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.conv5_1 = BaseBlockRelu(128, 256, 3, stride=2, padding=1, bias=False)

        self.res5a_b2a = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_b2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)

        self.res5b_b2a = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)
        self.res5b_b2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = x + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = x
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = x  # self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        x = self.conv5_1(x)
        x1 = x
        x2 = self.res5a_b2b(self.res5a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res5b_b2b(self.res5b_b2a(x))
        x = F.relu_(x + x1)

        return [x]

class Densebox_Net_32T_WEIMA1(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(Densebox_Net_32T_WEIMA1, self).__init__()
        self.conv1_1 = BaseBlockRelu(3, 32, 3, stride=2, padding=1, bias=False)

        self.conv2_1 = BaseBlockRelu(32, 32, 3, stride=2, padding=1, bias=False)
        self.res2a_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res2a_relu = nn.ReLU(inplace=False)            #wong

        self.res2b_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res2b_relu = nn.ReLU(inplace=False)            #wong

        self.conv3_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)

        # self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res3a_relu = nn.ReLU(inplace=False)            #wong

        self.res3b_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res3b_relu = nn.ReLU(inplace=False)            #wong

        self.conv4_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)

        # self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4a_relu = nn.ReLU(inplace=False)            #wong

        self.res4b_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4b_relu = nn.ReLU(inplace=False)            #wong

        self.res4c_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4c_relu = nn.ReLU(inplace=False)            #wong


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
        x = self.conv2_1(x)

        # x = x + self.res2a_b2b(self.res2a_b2a(x))
        # x = F.relu_(x)
        x = self.res2a_ew_add(x, self.res2a_b2b(self.res2a_b2a(x)))
        x = self.res2a_relu(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        # x = F.relu_(x2 + x)
        x = self.res2b_ew_add(x2, x)
        x = self.res2b_relu(x)

        x = self.conv3_1(x)
        x1 = x
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        # x = F.relu_(x1 + x2)
        x = self.res3a_ew_add(x1, x2)
        x = self.res3a_relu(x)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x = self.conv4_1(x)
        x1 = x  # self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        # x = F.relu_(x1 + x2)
        x = self.res4a_ew_add(x1, x2)
        x = self.res4a_relu(x)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        # x = F.relu_(x + x1)
        x = self.res4b_ew_add(x, x1)
        x = self.res4b_relu(x)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        # x = F.relu_(x + x1)
        x = self.res4c_ew_add(x, x1)
        x = self.res4c_relu(x)

        return [x]


class Densebox_Net_32T_WEIMA(nn.Module):
    def __init__(self, cfg):
        """
        Change num channels on top of Densebox_Net_NoBrranchConv
        All kernel change to 3x3
        """
        super(Densebox_Net_32T_WEIMA, self).__init__()
        self.conv1_1 = BaseBlockRelu(3, 16, 3, stride=2, padding=1, bias=False)
        self.conv1_2 = BaseBlockRelu(16, 16, 3, stride=1, padding=1, bias=False)

        self.conv2_1 = BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)
        self.res2a_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res2a_relu = nn.ReLU(inplace=False)            #wong

        self.res2b_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res2b_relu = nn.ReLU(inplace=False)            #wong

        self.conv3_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)

        # self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res3a_relu = nn.ReLU(inplace=False)            #wong

        self.res3b_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res3b_relu = nn.ReLU(inplace=False)            #wong

        self.conv4_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)

        # self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4a_relu = nn.ReLU(inplace=False)            #wong

        self.res4b_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4b_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4b_relu = nn.ReLU(inplace=False)            #wong

        self.res4c_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4c_ew_add = EltwiseAdd(inplace=False)       #wong
        self.res4c_relu = nn.ReLU(inplace=False)            #wong


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

        x = self.conv1_2(x)
        x = self.conv2_1(x)

        # x = x + self.res2a_b2b(self.res2a_b2a(x))
        # x = F.relu_(x)
        x = self.res2a_ew_add(x, self.res2a_b2b(self.res2a_b2a(x)))
        x = self.res2a_relu(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        # x = F.relu_(x2 + x)
        x = self.res2b_ew_add(x2, x)
        x = self.res2b_relu(x)

        x = self.conv3_1(x)
        x1 = x
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        # x = F.relu_(x1 + x2)
        x = self.res3a_ew_add(x1, x2)
        x = self.res3a_relu(x)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        # x = F.relu_(x + x1)
        x = self.res3b_ew_add(x, x1)
        x = self.res3b_relu(x)

        x = self.conv4_1(x)
        x1 = x  # self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        # x = F.relu_(x1 + x2)
        x = self.res4a_ew_add(x1, x2)
        x = self.res4a_relu(x)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        # x = F.relu_(x + x1)
        x = self.res4b_ew_add(x, x1)
        x = self.res4b_relu(x)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        # x = F.relu_(x + x1)
        x = self.res4c_ew_add(x, x1)
        x = self.res4c_relu(x)

        return [x]


class Densebox_Net_DW(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
        """
        super(Densebox_Net_DW, self).__init__()
        self.conv1 = BaseBlockRelu(3, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockDW(32, 32, 2, relu=True)  # BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)

        self.conv2_2 = BaseBlockDW(32, 16, 1, relu=True)  # BaseBlockRelu(32, 16, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockDW(16, 32, 1, relu=True)  # BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)

        self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlockDW(32, 32, 1, relu=False)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockDW(32, 64, 2, relu=True)  # BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockDW(64, 32, 1, relu=True)  # BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockDW(32, 64, 1, relu=True)  # BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockDW(64, 128, 2, relu=True)  # BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockDW(128, 64, 1, relu=True)  # BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockDW(64, 128, 1, relu=True)  # BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlockDW(128, 128, 1, relu=False)  # BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4b_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4c_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4c_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4d_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4d_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.res2a_b1(x) + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = self.res3a_b1(x)
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        return [res4d]


class Densebox_Net_DW_v2(nn.Module):
    def __init__(self, cfg):
        """
        As Densebox_Net_DW, conv2_1 DW block to base conv block
        """
        super(Densebox_Net_DW_v2, self).__init__()
        self.conv1 = BaseBlockRelu(3, 16, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)

        self.conv2_2 = BaseBlockDW(32, 16, 1, relu=True)  # BaseBlockRelu(32, 16, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockDW(16, 32, 1, relu=True)  # BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)

        self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockDW(32, 32, 1, relu=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlockDW(32, 32, 1, relu=False)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockDW(32, 64, 2, relu=True)  # BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockDW(64, 32, 1, relu=True)  # BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockDW(32, 64, 1, relu=True)  # BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockDW(64, 64, 1, relu=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlockDW(64, 64, 1, relu=False)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockDW(64, 128, 2, relu=True)  # BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockDW(128, 64, 1, relu=True)  # BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockDW(64, 128, 1, relu=True)  # BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlockDW(128, 128, 1, relu=False)  # BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4b_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4c_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4c_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4d_b2a = BaseBlockDW(128, 128, 1,
                                     relu=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4d_b2b = BaseBlockDW(128, 128, 1,
                                     relu=False)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.res2a_b1(x) + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = self.res3a_b1(x)
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        return [res4d]


class Densebox_Net_LinearDW(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
        """
        super(Densebox_Net_LinearDW, self).__init__()
        self.conv1 = BaseBlockRelu(3, 16, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockDW(16, 32, 2, relu=True,
                                   linear=True)  # BaseBlockRelu(16, 32, 3, stride=2, padding=1, bias=False)

        self.conv2_2 = BaseBlockDW(32, 16, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 16, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockDW(16, 32, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)

        self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockDW(32, 32, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlockDW(32, 32, 1, relu=True,
                                     linear=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockDW(32, 32, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlockDW(32, 32, 1, relu=False,
                                     linear=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockDW(32, 64, 2, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockDW(64, 32, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockDW(32, 64, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockDW(64, 64, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlockDW(64, 64, 1, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockDW(64, 64, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlockDW(64, 64, 1, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockDW(64, 64, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlockDW(64, 64, 1, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockDW(64, 128, 2, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockDW(128, 64, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockDW(64, 128, 1, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockDW(128, 128, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlockDW(128, 128, 1, relu=False,
                                     linear=True)  # BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockDW(128, 128, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4b_b2b = BaseBlockDW(128, 128, 1, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4c_b2a = BaseBlockDW(128, 128, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4c_b2b = BaseBlockDW(128, 128, 1, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4d_b2a = BaseBlockDW(128, 128, 1, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4d_b2b = BaseBlockDW(128, 128, 1, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.res2a_b1(x) + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = self.res3a_b1(x)
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        return [res4d]


class Densebox_Net_LinearDW5x5(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
        """
        super(Densebox_Net_LinearDW5x5, self).__init__()
        self.conv1 = BaseBlockRelu(3, 32, 5, stride=2, padding=2, bias=False)
        self.conv2_1 = BaseBlockRelu(32, 32, 5, stride=2, padding=2, bias=False)

        self.conv2_2 = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 16, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(16, 32, 3, stride=1, padding=1, bias=False)

        self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlockDW(32, 32, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockDW(32, 64, 2, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockDW(64, 32, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockDW(32, 64, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlockDW(64, 64, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockDW(64, 128, 2, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockDW(128, 64, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockDW(64, 128, 1, kernel=5, padding=2, relu=True,
                                   linear=True)  # BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4b_b2b = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4c_b2a = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4c_b2b = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4d_b2a = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=True,
                                     linear=True)  # BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4d_b2b = BaseBlockDW(128, 128, 1, kernel=5, padding=2, relu=False,
                                     linear=True)  # BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.res2a_b1(x) + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = self.res3a_b1(x)
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        return [res4d]


class BaseBlockDW(nn.Module):
    def __init__(self, inp, oup, stride, kernel=3, padding=1, linear=False, relu=True):
        super(BaseBlockDW, self).__init__()
        if linear:
            stage1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False), )
        else:
            stage1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True), )

        stage2 = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup), )
        if relu:
            self.model = nn.Sequential(stage1, stage2, nn.ReLU(inplace=True), )
        else:
            self.model = nn.Sequential(stage1, stage2, )

    def forward(self, x):
        return self.model(x)


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
