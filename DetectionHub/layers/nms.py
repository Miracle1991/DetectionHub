# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from DetectionHub import _C
import torch

# we need this for the custom ops to exist
import DetectionHub._custom_ops   # noqa: F401

from DetectionHub.utils import amp

_nms = torch.ops.DetectionHub.nms

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
