# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from DetectionHub.modeling import registry
from DetectionHub.modeling.backbone import resnet
from DetectionHub.modeling.poolers import Pooler
from DetectionHub.modeling.make_layers import group_norm
from DetectionHub.modeling.make_layers import make_fc


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("DenseboxRFCNROIFeatureExtractor32T")
class DenseboxRFCNROIFeatureExtractor32T(nn.Module):
    def __init__(self, config, in_channels):
        super(DenseboxRFCNROIFeatureExtractor32T, self).__init__()
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.class_pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            num_classes=num_classes,
            pooler_type=config.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        )

        in_channels = config.MODEL.BACKBONE.OUT_CHANNELS
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_classes = num_classes
        pooled_size = resolution * resolution

        self.res5a_b1 = BaseBlock(in_channels, 256, 1, stride=1, padding=0, bias=False)
        self.res5a_b2a = BaseBlockRelu(in_channels, 256, 3, stride=1, padding=1, bias=False)
        self.res5a_b2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)

        self.res5b_b2a = BaseBlockRelu(256, 256, 3, stride=1, padding=1, bias=False)
        self.res5b_b2b = BaseBlock(256, 256, 3, stride=1, padding=1, bias=False)

        self.conv_new_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.conv_new_1_relu = nn.ReLU(inplace=False)  # wong

        conv_cls_out_channel = num_classes * pooled_size

        self.conv_class = nn.Conv2d(256, conv_cls_out_channel, kernel_size=1, stride=1)
        self.out_channels = conv_cls_out_channel
        # if config.MODEL.ROI_BOX_HEAD.CLASS_AGNOSTIC:
        num_ag_cls = 2
        self.conv_regression = nn.Conv2d(256, num_ag_cls * pooled_size * 4, kernel_size=1, stride=1)

        self.regression_pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            num_classes=num_ag_cls,
            pooler_type=config.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        )
        # else:
        #     self.conv_regression = nn.Conv2d(conv_channels, num_classes * pooled_size * 4, kernel_size=1, stride=1)
        #     self.regression_pooler = Pooler(
        #         output_size=(resolution, resolution),
        #         scales=scales,
        #         sampling_ratio=sampling_ratio,
        #         num_classes=config.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        #         pooler_type=config.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals=None):
        input = x[0]
        x1 = self.res5a_b1(input)
        x2 = self.res5a_b2b(self.res5a_b2a(input))
        x = x1 + x2

        x1 = self.res5b_b2b(self.res5b_b2a(x))
        x = x1 + x
        res5b = F.relu_(x)

        x = self.conv_new_1(res5b)
        conv_new1 = F.relu_(x)

        class_features = self.conv_class(conv_new1)
        regression_features = self.conv_regression(conv_new1)

        if proposals is None:
            return [class_features, regression_features, conv_new1]
        else:
            regression_features = self.regression_pooler([regression_features], proposals)
            class_features = self.class_pooler([class_features], proposals)
            return [class_features, regression_features]


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


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

def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
