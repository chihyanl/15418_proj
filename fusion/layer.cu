
#include <cuda.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include "layer.h"

#define SEED 0
#define BLOCK_SIZE 256

int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__device__ __inline__ void cudaMax(float* temp, int N, int idx) {
  for (int twod = N/2; twod >= 1; twod /= 2) {
    if (idx < twod && temp[idx + twod] > temp[idx]) {
      temp[idx] = temp[idx + twod];
    }
    __syncthreads();
  }
}

__device__ __inline__ void cudaSum(float* temp, int N, int idx) {
  for (int twod = N/2; twod >= 1; twod /= 2) {
    if (idx < twod) {
      temp[idx] += temp[idx + twod];
    }
    __syncthreads();
  }
}

__global__ void cudaGetCorrect(int label, float* output, float* error, int N, int size, int* correct) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N) {
    return;
  }

  extern __shared__ float temp[];
  __shared__ int pred;

  temp[idx] = 0.0f;
  if (idx < size) {
    temp[idx] = output[idx];
    error[idx] = 0.0f;
  }
  
  if (idx == label) {
    error[idx] -= 1.0f;
  }
  __syncthreads();

  cudaMax(temp, N, idx);  // reduce max

  if (idx < size && temp[0] == output[idx]) {
    pred = idx;
  }
  __syncthreads();
  if (idx == 0) {
    *correct += pred == label;
  }
}

void getCorrect(int label, float* output, float* error, int size, int* correct) {
  int N = nextPow2(size);

  dim3 blockDim(N, 1);
  dim3 gridDim(1);
  cudaGetCorrect<<<gridDim, blockDim, sizeof(float) * N>>>(label, output, error, N, size, correct);
}

__global__ void randomFloat(float* ptr, float a, float b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size) {
    return;
  }

  curandState state;
  curand_init(SEED, idx, 0, &state);
  ptr[idx] = curand_uniform(&state) * (b - a) + a;
}

__global__ void cudaConvForward(int height, int width, int in_channels, int out_channels, int kernel_h, int kernel_w, int stride, int pad, int height_out, int width_out, float* input, float* output, float* weight, float* bias) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (outIdx >= out_channels * height_out * width_out) {
    return;
  }

  int dst_channel = outIdx / (height_out * width_out);
  int dst_y = (outIdx / width_out) % height_out;
  int dst_x = outIdx % width_out;

  int dstIdx_base = dst_channel * in_channels * kernel_h * kernel_w;
  int src_y = stride * dst_y - pad;
  int src_x = stride * dst_x - pad;

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
  
  sum = (sum > 0) ? sum : 0;  // relu
  output[outIdx] = sum;
}

__global__ void cudaConvBackward(int height, int width, int in_channels, int out_channels, int kernel_h, int kernel_w, int stride, int pad, int height_out, int width_out, float* input, float* output, float* weight, float* u_weight, float* u_bias, float* error, float* src_error) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (outIdx >= out_channels * height_out * width_out) {
    return;
  }

  int dst_channel = outIdx / (height_out * width_out);
  int dst_y = (outIdx / width_out) % height_out;
  int dst_x = outIdx % width_out;

  int dstIdx_base = dst_channel * in_channels * kernel_h * kernel_w;
  int src_y = stride * dst_y - pad;
  int src_x = stride * dst_x - pad;

  float dnet = error[outIdx] * (output[outIdx] > 0);
  atomicAdd(&u_bias[dst_channel], dnet);
  for (int src_channel = 0; src_channel < in_channels; src_channel++) {
    int dstIdx_base2 = src_channel * kernel_h * kernel_w;
    int srcIdx_base = src_channel * width * height;
    for (int dy = 0; dy < kernel_h; dy++) {
      int y = src_y + dy;
      if (y >= 0 && y < height) {
        int srcIdx = srcIdx_base + y * width;
        int dstIdx = dstIdx_base + dstIdx_base2 + dy * kernel_h;
        for (int dx = 0; dx < kernel_w; dx++) {
          int x = src_x + dx;
          if (x >= 0 && x < width) {
            if (src_error != nullptr) {
              atomicAdd(&src_error[srcIdx+x], weight[dstIdx+dx] * dnet);
            }
            atomicAdd(&u_weight[dstIdx+dx], dnet * input[srcIdx+x]);
          }
        }
      }
    }
  }
}

__global__ void cudaConvUpdate(int in_channels, int out_channels, int kernel_h, int kernel_w, float* weight, float* u_weight, float* bias, float* u_bias, float rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= out_channels * in_channels * kernel_h * kernel_w) {
    return;
  }

  weight[i] -= rate * u_weight[i];
  u_weight[i] = 0;

  if (i >= out_channels) {
    return;
  }

  bias[i] -= rate * u_bias[i];
  u_bias[i] = 0;
}

__global__ void cudaLinearForward(int in_channels, int out_channels, float* input, float* weight, float* bias, float* output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  extern __shared__ float temp[];

  float x = 0.0f;
  if (i < out_channels) {
    x = bias[i];
    for (int j = 0; j < in_channels; j++) {
      x += input[j] * weight[i*in_channels+j];
    }
    output[i] = x;
  }
  temp[i] = x;
  __syncthreads();

  // softmax 
  cudaMax(temp, N, i);  // reduce max

  x = 0.0f;
  if (i < out_channels) {
    x = expf(output[i] - temp[0]);
    output[i] = x;
  }
  temp[i] = x;
  __syncthreads();

  cudaSum(temp, N, i);  // reduce sum
  float sum = temp[0] + 0.0001; 
  
  //softmax
  if (i < out_channels) {
    output[i] /= sum;
  }
}

__global__ void cudaLinearBackward(int in_channels, int out_channels, float* input, float* weight, float* u_weight, float* u_bias, float* output, float* error, float* src_error) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= out_channels) {
    return;
  }

  float gradient = output[i] + error[i];
  for (int j = 0; j < in_channels; j++) {
    atomicAdd(&src_error[j], weight[i*in_channels+j] * gradient);
    u_weight[i*in_channels+j] += gradient * input[j];
  }
  u_bias[i] += gradient;
}

__global__ void cudaLinearUpdate(int in_channels, int out_channels, float* weight, float* u_weight, float* bias, float* u_bias, float* error, float rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= out_channels * in_channels) {
    return;
  }

  weight[i] -= rate * u_weight[i];
  u_weight[i] = 0.0f;

  if (i >= out_channels) {
    return;
  }

  bias[i] -= rate * u_bias[i];
  u_bias[i] = 0.0f;
}

Conv::Conv(int in_channels, int out_channels, int height, int width, int kernel_h, int kernel_w, int stride, int pad)
  : height(height),               // input height
    width(width),                 // input width
    in_channels(in_channels),     // number of input channels
    out_channels(out_channels),   // number of output channels
    kernel_h(kernel_h),           // filter height
    kernel_w(kernel_w),           // filter width
    stride(stride),
    pad(pad),
    height_out((height + 2 * pad - kernel_h) / stride + 1),
    width_out((width + 2 * pad - kernel_w) / stride + 1) {

  cudaMalloc(&output, sizeof(float) * out_channels * height_out * width_out);
  cudaMalloc(&bias, sizeof(float) * out_channels);
  cudaMalloc(&weight, sizeof(float) * out_channels * in_channels * kernel_h * kernel_w);

  cudaMemset(output, 0, sizeof(float) * out_channels * height_out * width_out);
  cudaMemset(bias, 0, sizeof(float) * out_channels);
  cudaMemset(weight, 0, sizeof(float) * out_channels * in_channels * kernel_h * kernel_w);

  cudaMalloc(&u_bias, sizeof(float) * out_channels);
  cudaMalloc(&u_weight, sizeof(float) * out_channels * in_channels * kernel_h * kernel_w);
  cudaMalloc(&error, sizeof(float) * out_channels * height_out * width_out);
  
  cudaMemset(u_bias, 0, sizeof(float) * out_channels);
  cudaMemset(u_weight, 0, sizeof(float) * out_channels * in_channels * kernel_h * kernel_w);
  cudaMemset(error, 0, sizeof(float) * out_channels * height_out * width_out);

  printf("Conv=%dx%dx%d (%d->%d), Kernel=%dx%d, Stride=%d, Pad=%d, Weight=%d, Bias=%d\n", height_out, width_out, out_channels, in_channels*width*height, out_channels*height_out*width_out, kernel_h, kernel_w, stride, pad, out_channels*in_channels*kernel_h*kernel_w, out_channels);

  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((out_channels * in_channels * kernel_h * kernel_w + blockDim.x - 1) / blockDim.x);
  
  float k = sqrt(1.0f / (in_channels * kernel_h * kernel_w));
  randomFloat<<<gridDim, blockDim>>>(weight, -k, k, out_channels * in_channels * kernel_h * kernel_w);
  randomFloat<<<gridDim, blockDim>>>(bias, -k, k, out_channels);
}

Conv::~Conv() {
  cudaFree(output);
  cudaFree(bias);
  cudaFree(weight);
  cudaFree(error);
  cudaFree(u_bias);
  cudaFree(u_weight);
}

void Conv::forward(float* input) {
  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((out_channels * height_out * width_out + blockDim.x - 1) / blockDim.x);

  cudaConvForward<<<gridDim, blockDim>>>(height, width, in_channels, out_channels, kernel_h, kernel_w, stride, pad, height_out, width_out, input, output, weight, bias);
}

void Conv::backward(float* input, float* src_error) {
  if (src_error != nullptr) {
    cudaMemset(src_error, 0, sizeof(float) * in_channels * width * height);
  }

  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((out_channels * height_out * width_out + blockDim.x - 1) / blockDim.x);

  cudaConvBackward<<<gridDim, blockDim>>>(height, width, in_channels, out_channels, kernel_h, kernel_w, stride, pad, height_out, width_out, input, output, weight, u_weight, u_bias, error, src_error);
}

void Conv::update(float rate) {
  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((out_channels * in_channels * kernel_h * kernel_w + blockDim.x - 1) / blockDim.x);

  cudaConvUpdate<<<gridDim, blockDim>>>(in_channels, out_channels, kernel_h, kernel_w, weight, u_weight, bias, u_bias, rate);
}


Linear::Linear(int in_channels, int out_channels)
  : in_channels(in_channels),
    out_channels(out_channels) {

  cudaMalloc(&output, sizeof(float) * out_channels);
  cudaMalloc(&bias, sizeof(float) * out_channels);
  cudaMalloc(&weight, sizeof(float) * in_channels * out_channels);

  cudaMemset(output, 0, sizeof(float) * out_channels);
  cudaMemset(bias, 0, sizeof(float) * out_channels);
  cudaMemset(weight, 0, sizeof(float) * in_channels * out_channels);

  cudaMalloc(&error, sizeof(float) * out_channels);
  cudaMalloc(&u_bias, sizeof(float) * out_channels);
  cudaMalloc(&u_weight, sizeof(float) * out_channels * in_channels);

  cudaMemset(error, 0, sizeof(float) * out_channels);
  cudaMemset(u_bias, 0, sizeof(float) * out_channels);
  cudaMemset(u_weight, 0, sizeof(float) * out_channels * in_channels);

  printf("Linear=%d->%d\n", in_channels, out_channels);
  
  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((out_channels * in_channels + blockDim.x - 1) / blockDim.x);

  float k = 1.0f / in_channels;
  randomFloat<<<gridDim, blockDim>>>(weight, -k, k, out_channels * in_channels);
  randomFloat<<<gridDim, blockDim>>>(bias, -k, k, out_channels);
}

Linear::~Linear() {
  cudaFree(output);
  cudaFree(bias);
  cudaFree(weight);
  cudaFree(error);
  cudaFree(u_bias);
  cudaFree(u_weight);
}

void Linear::forward(float* input) {
  int N = nextPow2(out_channels);

  if (N > BLOCK_SIZE) {
    printf("Error: Linear Layer out_channels greater than BLOCK_SIZE");
    return;
  }

  dim3 blockDim(N, 1);
  dim3 gridDim(1);
  cudaLinearForward<<<gridDim, blockDim, sizeof(float) * N>>>(in_channels, out_channels, input, weight, bias, output, N);
}

void Linear::backward(float* input, float* src_error) {
  cudaMemset(src_error, 0, sizeof(float) * in_channels);

  dim3 blockDim(out_channels, 1);
  dim3 gridDim(1);
  cudaLinearBackward<<<gridDim, blockDim>>>(in_channels, out_channels, input, weight, u_weight, u_bias, output, error, src_error);
}

void Linear::update(float rate) {
  dim3 blockDim(BLOCK_SIZE, 1);
  dim3 gridDim((in_channels * out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
  cudaLinearUpdate<<<gridDim, blockDim>>>(in_channels, out_channels, weight, u_weight, bias, u_bias, error, rate);
}
