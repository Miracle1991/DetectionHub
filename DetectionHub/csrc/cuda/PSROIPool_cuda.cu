// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template<typename T>
__global__ void PSROIPoolFForward(const int nthreads, const T *bottom_data_ori,
                                  const float spatial_scale, const int height, const int width,
                                  const int channels, const int pooled_height,
                                  const int pooled_width,
                                  const int group_size, const int output_dim,
                                  const T *offset_bottom_rois, T *top_data, int *mapping_channel) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        const T *bottom_rois = offset_bottom_rois + n * 5;
        //bottom_rois += n * 5;
        int roi_batch_ind = static_cast<int>(bottom_rois[0]);
        //if(roi_batch_ind>19) printf("roi batch ind error %d %d\n", roi_batch_ind, n);
        T roi_start_w =
                static_cast<T>(round(bottom_rois[1])) * spatial_scale;
        T roi_start_h =
                static_cast<T>(round(bottom_rois[2])) * spatial_scale;
        T roi_end_w =
                static_cast<T>(round(bottom_rois[3]) + 1.) * spatial_scale;
        T roi_end_h =
                static_cast<T>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force malformed ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        T roi_height = max(roi_end_h - roi_start_h, 0.1);

        T bin_size_h = (T) (roi_height) / (T) (pooled_height);
        T bin_size_w = (T) (roi_width) / (T) (pooled_width);

        int hstart = floor(static_cast<T>(ph) * bin_size_h
                           + roi_start_h);
        int wstart = floor(static_cast<T>(pw) * bin_size_w
                           + roi_start_w);
        int hend = ceil(static_cast<T>(ph + 1) * bin_size_h
                        + roi_start_h);
        int wend = ceil(static_cast<T>(pw + 1) * bin_size_w
                        + roi_start_w);

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        const T *bottom_data = bottom_data_ori + (roi_batch_ind * channels + c) * height * width;
        //bottom_data += (roi_batch_ind * channels + c) * height * width;
        T out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                out_sum += bottom_data[bottom_index];
            }
        }
        T bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? 0. : out_sum / bin_area;
        mapping_channel[index] = c;
    }
}

template<typename T>
__global__ void PSROIPoolBackward(const int nthreads, const T *top_diff,
                                  const int *mapping_channel, const int num_rois,
                                  const float spatial_scale,
                                  const int height, const int width, const int channels,
                                  const int pooled_height, const int pooled_width,
                                  const int output_dim, T *bottom_diff,
                                  const T *bottom_rois_ori) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {

        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const T *bottom_rois = bottom_rois_ori + n * 5;
        //bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        T roi_start_w =
                static_cast<T>(round(bottom_rois[1])) * spatial_scale;
        T roi_start_h =
                static_cast<T>(round(bottom_rois[2])) * spatial_scale;
        T roi_end_w =
                static_cast<T>(round(bottom_rois[3]) + 1.) * spatial_scale;
        T roi_end_h =
                static_cast<T>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        T roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        int hstart = floor(static_cast<T>(ph) * bin_size_h
                           + roi_start_h);
        int wstart = floor(static_cast<T>(pw) * bin_size_w
                           + roi_start_w);
        int hend = ceil(static_cast<T>(ph + 1) * bin_size_h
                        + roi_start_h);
        int wend = ceil(static_cast<T>(pw + 1) * bin_size_w
                        + roi_start_w);
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Compute c at bottom
        int c = mapping_channel[index];
        T *offset_bottom_diff = bottom_diff +
                                (roi_batch_ind * channels + c) * height * width;
        T bin_area = (hend - hstart) * (wend - wstart);
        T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
                atomicAdd(offset_bottom_diff + bottom_index, diff_val);
            }
        }
    }
}

at::Tensor PSROIPool_forward_cuda(const at::Tensor &input,
                                  const at::Tensor &rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  at::Tensor &mapping_channel) {
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

    auto num_rois = rois.size(0);
    int size_rois = rois.size(1);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto group_size = pooled_height * pooled_width;
    int output_dim = channels / group_size; //the num of channels in output
    if (size_rois != 5) {
        std::cout << "rois size is not equal to 5!" << std::endl;
    }
    if (channels % group_size) {
        std::cout << "the size of channels must be times of group_size" << std::endl;
    }

    auto output = at::empty({num_rois, output_dim, pooled_height, pooled_width}, input.options());

    auto output_size = num_rois * pooled_height * pooled_width * output_dim;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    //std::cout<<"compare 4096: "<<THCCeilDiv(output_size, 512L)<<std::endl;
    dim3
    grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
    dim3 block(512);
    //std::cout<<"types: "<<input.type()<<" "<<output.type()<<" "<<mapping_channel.type()<<std::endl;
    if (output.numel() == 0) {
        //std::cout<<"first place\n";
        THCudaCheck(cudaGetLastError());
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "PSROIPool_forward", [&] {
        PSROIPoolFForward<scalar_t> << < grid, block, 0, stream >> > (
                output_size,
                        input.contiguous().data<scalar_t>(),
                        spatial_scale,
                        height,
                        width,
                        channels,
                        pooled_height,
                        pooled_width,
                        pooled_height,
                        output_dim,
                        rois.contiguous().data<scalar_t>(),
                        output.data<scalar_t>(),
                        mapping_channel.contiguous().data<int>());
    });
    //std::cout<<"second place\n";
    THCudaCheck(cudaGetLastError());
    //std::cout<<"third place\n";
    return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor PSROIPool_backward_cuda(const at::Tensor &grad,
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
    AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
    // TODO add more checks

    auto num_rois = rois.size(0);
    int size_rois = rois.size(1);
    auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());
    auto group_size = pooled_height * pooled_width;
    int output_dim = channels / group_size; //the num of channels in output
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3
    grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));
    dim3 block(512);

    // handle possibly empty gradients
    if (grad.numel() == 0) {
        //std::cout<<"backward first place\n";
        THCudaCheck(cudaGetLastError());
        return grad_input;
    }

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "PSROIPool_backward", [&] {
        PSROIPoolBackward<scalar_t> << < grid, block, 0, stream >> > (
                output_size,
                        grad.contiguous().data<scalar_t>(),
                        mappingchannel.contiguous().data<int>(),
                        num_rois,
                        spatial_scale,
                        height,
                        width,
                        channels,
                        pooled_height,
                        pooled_width,
                        output_dim,
                        grad_input.data<scalar_t>(),
                        rois.contiguous().data<scalar_t>());
    });
    //std::cout<<"backward second place\n";
    THCudaCheck(cudaGetLastError());
    //std::cout<<"backward third place\n";
    return grad_input;
}
