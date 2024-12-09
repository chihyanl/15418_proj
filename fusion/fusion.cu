
#include <cuda.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <iostream>

#include "layer.h"
#define BLOCK_SIZE_MAX 1024

__device__ __inline__ void cudaCopyInput(float* src, float* dst, int in_channels, int height, int width, int tile_h, int tile_w, int src_x_base, int src_y_base) {

  for (int i = threadIdx.x; i < in_channels * tile_h * tile_w; i += blockDim.x) {
    int channel = i / (tile_w * tile_h);
    int dst_y = (i / tile_w) % tile_h;
    int dst_x = i % tile_w;
           
    int src_x = src_x_base + dst_x;
    int src_y = src_y_base + dst_y;
    dst[i] = 0;
    if (src_y >= 0 && src_y < height && src_x >= 0 && src_x < width) {
      int src_idx = channel * width * height + src_y * width + src_x;
      dst[i] = src[src_idx];
    }
  }
}

__device__ __inline__ void cudaCopyOutput(float* src, float* dst, int out_channels, int height, int width, int tile_h, int tile_w) {

  if (threadIdx.x >= out_channels * tile_h * tile_w) {
    return;
  }

  int channel = threadIdx.x / (tile_w * tile_h);
  int src_y = (threadIdx.x / tile_w) % tile_h;
  int src_x = threadIdx.x % tile_w;

  int dst_x = blockIdx.x * tile_w + src_x;
  int dst_y = blockIdx.y * tile_h + src_y;
  if (dst_y >= 0 && dst_y < height && dst_x >= 0 && dst_x < width) {
    int dst_idx = channel * width * height + dst_y * width + dst_x;
    dst[dst_idx] = src[threadIdx.x];
  }
}

__device__ __inline__ void cudaConvFuseForward(int height, int width, int in_channels, int out_channels, int kernel_h, int kernel_w, int stride, int height_out, int width_out, float* input, float* output, float* weight, float* bias) {

  if (threadIdx.x >= out_channels * width_out * height_out) {
    return;
  }

  int outIdx = threadIdx.x;

  int dst_channel = outIdx / (width_out * height_out);
  int dst_y = (outIdx / width_out) % height_out;
  int dst_x = outIdx % width_out;

  int dstIdx_base = dst_channel * in_channels * kernel_h * kernel_w;
  int src_y = stride * dst_y;
  int src_x = stride * dst_x;

  float sum = bias[dst_channel];
  for (int src_channel = 0; src_channel < in_channels; src_channel++) {
    int dstIdx_base2 = src_channel * kernel_h * kernel_w;
    int srcIdx_base = src_channel * width * height;
    for (int dy = 0; dy < kernel_h; dy++) {
      int y = src_y + dy;
      if (y >= 0 && y < height) {
        int srcIdx = srcIdx_base + y * width;
        int dstIdx = dstIdx_base + dstIdx_base2 + dy * kernel_w;
        for (int dx = 0; dx < kernel_w; dx++) {
          int x = src_x + dx;
          if (x >= 0 && x < width) {
            sum += input[srcIdx+x] * weight[dstIdx+dx];
          }
        }
      }
    }
  }
  sum = (sum > 0) ? sum : 0;
  output[outIdx] = sum;
}

__global__ void cudaConvFuseForward2(int height1, int width1, int channels1, int kernel_h1, int kernel_w1, int stride1, int pad1,
                                 int height2, int width2, int channels2, int kernel_h2, int kernel_w2, int stride2, int pad2,
                                 int height3, int width3, int channels3,
                                 int tile_h1, int tile_w1, int tile_h2, int tile_w2, int tile_h3, int tile_w3,
                                 float* input, float* output, float* weight1, float* bias1, float* weight2, float* bias2) {
  extern __shared__ float array[];
  float* bufI = array;
  float* bufInter = &array[channels1 * tile_h1 * tile_w1];
  float* bufO = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2];

  int in_x_base = (blockIdx.x * tile_w3 * stride2 - pad2) * stride1 - pad1;
  int in_y_base = (blockIdx.y * tile_h3 * stride2 - pad2) * stride1 - pad1;

  cudaCopyInput(input, bufI, channels1, height1, width1, tile_h1, tile_w1, in_x_base, in_y_base);
  __syncthreads();

  cudaConvFuseForward(tile_h1, tile_w1, channels1, channels2, kernel_h1, kernel_w1, stride1, tile_h2, tile_w2, bufI, bufInter, weight1, bias1);
  __syncthreads();

  cudaConvFuseForward(tile_h2, tile_w2, channels2, channels3, kernel_h2, kernel_w2, stride2, tile_h3, tile_w3, bufInter, bufO, weight2, bias2);
  __syncthreads();

  cudaCopyOutput(bufO, output, channels3, height3, width3, tile_h3, tile_w3);
}

__global__ void cudaConvFuseForward3(int height1, int width1, int channels1, int kernel_h1, int kernel_w1, int stride1, int pad1,
                                 int height2, int width2, int channels2, int kernel_h2, int kernel_w2, int stride2, int pad2,
                                 int height3, int width3, int channels3, int kernel_h3, int kernel_w3, int stride3, int pad3,
                                 int height4, int width4, int channels4,
                                 int tile_h1, int tile_w1, int tile_h2, int tile_w2, int tile_h3, int tile_w3, int tile_h4, int tile_w4,
                                 float* input, float* output, float* weight1, float* bias1, float* weight2, float* bias2, float* weight3, float* bias3) {
  extern __shared__ float array[];
  float* bufI = array;
  float* bufInter1 = &array[channels1 * tile_h1 * tile_w1];
  float* bufInter2 = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2];
  float* bufO = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2 + channels3 * tile_h3 * tile_w3];

  int in_x_base = ((blockIdx.x * tile_w4 * stride3 - pad3) * stride2 - pad2) * stride1 - pad1;
  int in_y_base = ((blockIdx.y * tile_h4 * stride3 - pad3) * stride2 - pad2) * stride1 - pad1;

  cudaCopyInput(input, bufI, channels1, height1, width1, tile_h1, tile_w1, in_x_base, in_y_base);
  __syncthreads();

  cudaConvFuseForward(tile_h1, tile_w1, channels1, channels2, kernel_h1, kernel_w1, stride1, tile_h2, tile_w2, bufI, bufInter1, weight1, bias1);
  __syncthreads();

  cudaConvFuseForward(tile_h2, tile_w2, channels2, channels3, kernel_h2, kernel_w2, stride2, tile_h3, tile_w3, bufInter1, bufInter2, weight2, bias2);
  __syncthreads();
  
  cudaConvFuseForward(tile_h3, tile_w3, channels3, channels4, kernel_h3, kernel_w3, stride3, tile_h4, tile_w4, bufInter2, bufO, weight3, bias3);
  __syncthreads();

  cudaCopyOutput(bufO, output, channels4, height4, width4, tile_h4, tile_w4);
}

__global__ void cudaConvFuseForward4(int height1, int width1, int channels1, int kernel_h1, int kernel_w1, int stride1, int pad1,
                                 int height2, int width2, int channels2, int kernel_h2, int kernel_w2, int stride2, int pad2,
                                 int height3, int width3, int channels3, int kernel_h3, int kernel_w3, int stride3, int pad3,
                                 int height4, int width4, int channels4, int kernel_h4, int kernel_w4, int stride4, int pad4,
                                 int height5, int width5, int channels5,
                                 int tile_h1, int tile_w1, int tile_h2, int tile_w2, int tile_h3, int tile_w3, int tile_h4, int tile_w4, int tile_h5, int tile_w5,
                                 float* input, float* output, float* weight1, float* bias1, float* weight2, float* bias2, float* weight3, float* bias3, float* weight4, float* bias4) {
  extern __shared__ float array[];
  float* bufI = array;
  float* bufInter1 = &array[channels1 * tile_h1 * tile_w1];
  float* bufInter2 = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2];
  float* bufInter3 = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2 + channels3 * tile_h3 * tile_w3];
  float* bufO = &array[channels1 * tile_h1 * tile_w1 + channels2 * tile_h2 * tile_w2 + channels3 * tile_h3 * tile_w3 + channels4 * tile_h4 * tile_w4];

  int in_x_base = (((blockIdx.x * tile_w5 * stride4 - pad4) * stride3 - pad3) * stride2 - pad2) * stride1 - pad1;
  int in_y_base = (((blockIdx.y * tile_h5 * stride4 - pad4) * stride3 - pad3) * stride2 - pad2) * stride1 - pad1;

  cudaCopyInput(input, bufI, channels1, height1, width1, tile_h1, tile_w1, in_x_base, in_y_base);
  __syncthreads();

  cudaConvFuseForward(tile_h1, tile_w1, channels1, channels2, kernel_h1, kernel_w1, stride1, tile_h2, tile_w2, bufI, bufInter1, weight1, bias1);
  __syncthreads();

  cudaConvFuseForward(tile_h2, tile_w2, channels2, channels3, kernel_h2, kernel_w2, stride2, tile_h3, tile_w3, bufInter1, bufInter2, weight2, bias2);
  __syncthreads();

  cudaConvFuseForward(tile_h3, tile_w3, channels3, channels4, kernel_h3, kernel_w3, stride3, tile_h4, tile_w4, bufInter2, bufInter3, weight3, bias3);
  __syncthreads();
  
  cudaConvFuseForward(tile_h4, tile_w4, channels4, channels4, kernel_h4, kernel_w4, stride4, tile_h5, tile_w5, bufInter3, bufO, weight4, bias4);
  __syncthreads();

  cudaCopyOutput(bufO, output, channels5, height5, width5, tile_h5, tile_w5);
}

void convFuseForward2(float* input, Conv* l1, Conv* l2, int tile_h, int tile_w) {

  int num_tiles_w = (l2->width_out + tile_w - 1) / tile_w;
  int num_tiles_h = (l2->height_out + tile_h - 1) / tile_h;

  int tile_h3 = tile_h;
  int tile_w3 = tile_w;
  int tile_h2 = (tile_h3 - 1) * l2->stride + l2->kernel_h;
  int tile_w2 = (tile_w3 - 1) * l2->stride + l2->kernel_w;
  int tile_h1 = (tile_h2 - 1) * l1->stride + l1->kernel_h;
  int tile_w1 = (tile_w2 - 1) * l1->stride + l1->kernel_w;
  int thread_count = max(l1->out_channels * tile_h2 * tile_w2, l2->out_channels * tile_h3 * tile_w3);

  if (thread_count > BLOCK_SIZE_MAX) {
    std::cerr << "ERROR: 2 layer fusion Tile_Width*Tile_Height*Out_Channels in any layer cannot exceed " << BLOCK_SIZE_MAX << "\n";
    return;
  }

  int bufI_size = l1->in_channels * tile_h1 * tile_w1;
  int bufInter_size = l2->in_channels * tile_h2 * tile_w2;
  int bufO_size = l2->out_channels * tile_h3 * tile_w3;
  int buffer_size = bufI_size + bufInter_size + bufO_size;

  dim3 blockDim(thread_count, 1);
  dim3 gridDim(num_tiles_w, num_tiles_h);

  cudaConvFuseForward2<<<gridDim, blockDim, sizeof(float) * buffer_size>>>
                                         (l1->height, l1->width, l1->in_channels, l1->kernel_h, l1->kernel_w, l1->stride, l1->pad,
                                          l2->height, l2->width, l2->in_channels, l2->kernel_h, l2->kernel_w, l2->stride, l2->pad,
                                          l2->height_out, l2->width_out, l2->out_channels,
                                          tile_h1, tile_w1, tile_h2, tile_w2, tile_h3, tile_w3,
                                          input, l2->output, l1->weight, l1->bias, l2->weight, l2->bias);
  cudaDeviceSynchronize();
}

void convFuseForward3(float* input, Conv* l1, Conv* l2, Conv* l3, int tile_h, int tile_w) {

  int num_tiles_w = (l3->width_out + tile_w - 1) / tile_w;
  int num_tiles_h = (l3->height_out + tile_h - 1) / tile_h;

  int tile_h4 = tile_h;
  int tile_w4 = tile_w;
  int tile_h3 = (tile_h4 - 1) * l3->stride + l3->kernel_h;
  int tile_w3 = (tile_w4 - 1) * l3->stride + l3->kernel_w;
  int tile_h2 = (tile_h3 - 1) * l2->stride + l2->kernel_h;
  int tile_w2 = (tile_w3 - 1) * l2->stride + l2->kernel_w;
  int tile_h1 = (tile_h2 - 1) * l1->stride + l1->kernel_h;
  int tile_w1 = (tile_w2 - 1) * l1->stride + l1->kernel_w;
  int thread_count = max(max(l1->out_channels * tile_h2 * tile_w2, l2->out_channels * tile_h3 * tile_w3), l3->out_channels * tile_h4 * tile_w4);

  if (thread_count > BLOCK_SIZE_MAX) {
    std::cerr << "ERROR: 3 layer fusion Tile_Width*Tile_Height*Out_Channels in any layer cannot exceed " << BLOCK_SIZE_MAX << "\n";
    return;
  }

  int bufI_size = l1->in_channels * tile_h1 * tile_w1;
  int bufInter_size1 = l2->in_channels * tile_h2 * tile_w2;
  int bufInter_size2 = l3->in_channels * tile_h3 * tile_w3;
  int bufO_size = l3->out_channels * tile_h4 * tile_w4;
  int buffer_size = bufI_size + bufInter_size1 + bufInter_size2 + bufO_size;

  dim3 blockDim(thread_count, 1);
  dim3 gridDim(num_tiles_w, num_tiles_h);

  cudaConvFuseForward3<<<gridDim, blockDim, sizeof(float) * buffer_size>>>
                                         (l1->height, l1->width, l1->in_channels, l1->kernel_h, l1->kernel_w, l1->stride, l1->pad,
                                          l2->height, l2->width, l2->in_channels, l2->kernel_h, l2->kernel_w, l2->stride, l2->pad,
                                          l3->height, l3->width, l3->in_channels, l3->kernel_h, l3->kernel_w, l3->stride, l3->pad,
                                          l3->height_out, l3->width_out, l3->out_channels,
                                          tile_h1, tile_w1, tile_h2, tile_w2, tile_h3, tile_w3, tile_h4, tile_w4,
                                          input, l3->output, l1->weight, l1->bias, l2->weight, l2->bias, l3->weight, l3->bias);
  cudaDeviceSynchronize();
}

void convFuseForward4(float* input, Conv* l1, Conv* l2, Conv* l3, Conv* l4, int tile_h, int tile_w) {

  int num_tiles_w = (l4->width_out + tile_w - 1) / tile_w;
  int num_tiles_h = (l4->height_out + tile_h - 1) / tile_h;

  int tile_h5 = tile_h;
  int tile_w5 = tile_w;
  int tile_h4 = (tile_h5 - 1) * l4->stride + l4->kernel_h;
  int tile_w4 = (tile_w5 - 1) * l4->stride + l4->kernel_w;
  int tile_h3 = (tile_h4 - 1) * l3->stride + l3->kernel_h;
  int tile_w3 = (tile_w4 - 1) * l3->stride + l3->kernel_w;
  int tile_h2 = (tile_h3 - 1) * l2->stride + l2->kernel_h;
  int tile_w2 = (tile_w3 - 1) * l2->stride + l2->kernel_w;
  int tile_h1 = (tile_h2 - 1) * l1->stride + l1->kernel_h;
  int tile_w1 = (tile_w2 - 1) * l1->stride + l1->kernel_w;
  int thread_count = max(max(max(l1->out_channels * tile_h2 * tile_w2, l2->out_channels * tile_h3 * tile_w3), l3->out_channels * tile_h4 * tile_w4), l4->out_channels * tile_h5 * tile_w5);

  if (thread_count > BLOCK_SIZE_MAX) {
    std::cerr << "ERROR: 4 layer fusion Tile_Width*Tile_Height*Out_Channels in any layer cannot exceed " << BLOCK_SIZE_MAX << "\n";
    return;
  }

  int bufI_size = l1->in_channels * tile_h1 * tile_w1;
  int bufInter_size1 = l2->in_channels * tile_h2 * tile_w2;
  int bufInter_size2 = l3->in_channels * tile_h3 * tile_w3;
  int bufInter_size3 = l4->in_channels * tile_h4 * tile_w4;
  int bufO_size = l4->out_channels * tile_h5 * tile_w5;
  int buffer_size = bufI_size + bufInter_size1 + bufInter_size2 + bufInter_size3 + bufO_size;

  dim3 blockDim(thread_count, 1);
  dim3 gridDim(num_tiles_w, num_tiles_h);

  cudaConvFuseForward4<<<gridDim, blockDim, sizeof(float) * buffer_size>>>
                                         (l1->height, l1->width, l1->in_channels, l1->kernel_h, l1->kernel_w, l1->stride, l1->pad,
                                          l2->height, l2->width, l2->in_channels, l2->kernel_h, l2->kernel_w, l2->stride, l2->pad,
                                          l3->height, l3->width, l3->in_channels, l3->kernel_h, l3->kernel_w, l3->stride, l3->pad,
                                          l4->height, l4->width, l4->in_channels, l4->kernel_h, l4->kernel_w, l4->stride, l4->pad,
                                          l4->height_out, l4->width_out, l4->out_channels,
                                          tile_h1, tile_w1, tile_h2, tile_w2, tile_h3, tile_w3, tile_h4, tile_w4, tile_h5, tile_w5,
                                          input, l4->output, l1->weight, l1->bias, l2->weight, l2->bias, l3->weight, l3->bias, l4->weight, l4->bias);
  cudaDeviceSynchronize();
}
