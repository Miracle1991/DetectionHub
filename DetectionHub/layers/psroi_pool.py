# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import DetectionHub._C as _C
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair


class _PSROIPool(Function):
    @staticmethod
    def forward(ctx, input, rois, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        input_channels = input.size()[1]
        pooled_size = ctx.output_size[0] * ctx.output_size[1]
        ctx.output_dim = int(input_channels / pooled_size)
        num_rois = rois.size()[0]
        device, dtype = input.device, input.dtype
        mapping_channel = torch.zeros(num_rois, ctx.output_dim, ctx.output_size[0], ctx.output_size[1], device=device,
                                      dtype=torch.int32)
        output = _C.psroi_pool_forward(input, rois, ctx.spatial_scale, ctx.output_size[0], ctx.output_size[1],
                                       mapping_channel)
        ctx.save_for_backward(input, mapping_channel, rois)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, mapping_channel, rois = ctx.saved_tensors
        batch_size, num_channels, data_height, data_width = input.size()
        grad_input = _C.psroi_pool_backward(grad_output, input, rois, ctx.spatial_scale, ctx.output_size[0],
                                            ctx.output_size[1], batch_size, num_channels, data_height, data_width,
                                            mapping_channel)
        return grad_input, None, None, None

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale):
        """
        as titled
        """
        return g.op('PSROIPool', input, rois, output_size_i=output_size, spatial_scale_f=spatial_scale)


psroi_pool = _PSROIPool.apply


class PSROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(PSROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return psroi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        # tmpstr += ", output_dim=" + str(self.output_dim)
        tmpstr += ")"
        return tmpstr
