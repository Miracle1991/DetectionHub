# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn
import torch

class Yolov3Predictor(nn.Module):
    def __init__(self, cfg):
        super(Yolov3Predictor, self).__init__()

    def forward(self, x):
        # N = x.shape[0]
        # C = x.shape[1]
        # H = x.shape[2]
        # W = x.shape[3]
        # x = x.permute(0, 2, 3, 1)
        # x = x.view(N, H, W, 3, int(C/3))

        # objectness = x[:, :, :, :, 0]
        # rpn_box_regression = x[:, :, :, :, 1:5]
        # cls = x[:, :, :, :, 5:85]

        objectness1 = x[:, 0, :, :]
        objectness2 = x[:, 85, :, :]
        objectness3 = x[:, 170, :, :]
        objectness1 = torch.unsqueeze(objectness1, 1)
        objectness2 = torch.unsqueeze(objectness2, 1)
        objectness3 = torch.unsqueeze(objectness3, 1)
        objectness = torch.cat((objectness1, objectness2, objectness3), 1)
        rpn_box_regression1 = x[:, 1:5, :, :]
        rpn_box_regression2 = x[:, 86:90, :, :]
        rpn_box_regression3 = x[:, 171:175, :, :]
        rpn_box_regression = torch.cat((rpn_box_regression1, rpn_box_regression2, rpn_box_regression3), 1)
        cls1 = x[:, 5:85, :, :]
        cls2 = x[:, 90:170, :, :]
        cls3 = x[:, 175:255, :, :]
        cls = torch.cat((cls1, cls2, cls3), 1)

        return [objectness, rpn_box_regression, cls]


def make_yolov3_predictor(cfg):
    return Yolov3Predictor(cfg)
