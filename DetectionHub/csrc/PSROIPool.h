// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor PSROIPool_forward(const at::Tensor &input,
                             const at::Tensor &rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             at::Tensor &mapping_channel) {
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return PSROIPool_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, mapping_channel);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor PSROIPool_backward(const at::Tensor &grad,
                              const at::Tensor &input,
                              const at::Tensor &rois,
                              const float spatial_scale,
                              const int pooled_height,
                              const int pooled_width,
                              const int batch_size,
                              const int channels,
                              const int height,
                              const int width,
                              const at::Tensor &mappingchannel) {
    if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
        return PSROIPool_backward_cuda(grad, input, rois, spatial_scale, pooled_height, pooled_width, batch_size,
        channels, height, width, mappingchannel);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}



