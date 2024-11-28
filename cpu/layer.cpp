#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include "layer.h"


float randomFloat(float a, float b) {
  float random_value = (float)rand() / RAND_MAX;
  return random_value * (b - a) + a;
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

  output = (float*)calloc(out_channels * height_out * width_out, sizeof(float));
  bias = (float*)calloc(out_channels, sizeof(float));
  weight = (float*)calloc(out_channels * in_channels * kernel_h * kernel_w, sizeof(float));

  u_bias = (float*)calloc(out_channels, sizeof(float));
  u_weight = (float*)calloc(out_channels * in_channels * kernel_h * kernel_w, sizeof(float));
  gradient = (float*)calloc(out_channels * height_out * width_out, sizeof(float));
  error = (float*)calloc(out_channels * height_out * width_out, sizeof(float));

  printf("Conv=%dx%dx%d (%d->%d), Kernel=%dx%d, Stride=%d, Pad=%d, Weight=%d, Bias=%d\n", height_out, width_out, out_channels, in_channels*width*height, out_channels*height_out*width_out, kernel_h, kernel_w, stride, pad, out_channels*in_channels*kernel_h*kernel_w, out_channels);

  float k = sqrt(1.0f / (in_channels * kernel_h * kernel_w));
  for (int i = 0; i < out_channels * in_channels * kernel_h * kernel_w; i++) {
    weight[i] = randomFloat(-k, k);
  }
  for (int i = 0; i < out_channels; i++) {
    bias[i] = randomFloat(-k, k);
  }
}

Conv::~Conv() {
  free(output);
  free(bias);
  free(weight);
  free(error);
  free(u_bias);
  free(u_weight);
  free(gradient);
}

void Conv::forward(float* input) {
  for (int dst_channel = 0; dst_channel < out_channels; dst_channel++) {
    int dstIdx_base = dst_channel * in_channels * kernel_h * kernel_w;
    int outIdx_base = dst_channel * height_out * width_out;
    for (int dst_y = 0; dst_y < height_out; dst_y++) {
      int src_y = stride * dst_y - pad;
      int outIdx_base2 = dst_y * width_out;
      for (int dst_x = 0; dst_x < width_out; dst_x++) {
        int src_x = stride * dst_x - pad;
        int outIdx = outIdx_base + outIdx_base2 + dst_x;
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
        gradient[outIdx] = (sum > 0) ? 1 : 0;
      }
    }
  }
}

void Conv::backward(float* input, float* src_error) {
  if (src_error != NULL) {
    for (int i = 0; i < in_channels * width * height; i++) {
      src_error[i] = 0;
    }
  }

  for (int dst_channel = 0; dst_channel < out_channels; dst_channel++) {
    int dstIdx_base = dst_channel * in_channels * kernel_h * kernel_w;
    int outIdx_base = dst_channel * height_out * width_out;
    for (int dst_y = 0; dst_y < height_out; dst_y++) {
      int src_y = stride * dst_y - pad;
      int outIdx_base2 = dst_y * width_out;
      for (int dst_x = 0; dst_x < width_out; dst_x++) {
        int src_x = stride * dst_x - pad;
        int outIdx = outIdx_base + outIdx_base2 + dst_x;
        float dnet = error[outIdx] * gradient[outIdx];
        u_bias[dst_channel] += dnet;
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
                  if (src_error != NULL) {
                    src_error[srcIdx+x] += weight[dstIdx+dx] * dnet;
                  }
                  u_weight[dstIdx+dx] += dnet * input[srcIdx+x];
                }
              }
            }
          }
        }
      }
    }
  }
}

void Conv::update(float rate) {
  for (int i = 0; i < out_channels; i++) {
    bias[i] -= rate * u_bias[i];
    u_bias[i] = 0;
  }
  for (int i = 0; i < out_channels * in_channels * kernel_h * kernel_w; i++) {
    weight[i] -= rate * u_weight[i];
    u_weight[i] = 0;
  }
}


Linear::Linear(int in_channels, int out_channels)
  : in_channels(in_channels),
    out_channels(out_channels) {

  output = (float*)calloc(out_channels, sizeof(float));
  bias = (float*)calloc(out_channels, sizeof(float));
  weight = (float*)calloc(in_channels * out_channels, sizeof(float));

  error = (float*)calloc(out_channels, sizeof(float));
  u_bias = (float*)calloc(out_channels, sizeof(float));
  u_weight = (float*)calloc(in_channels * out_channels, sizeof(float));
  gradient = (float*)calloc(out_channels, sizeof(float));
  
  printf("Linear=%d->%d\n", in_channels, out_channels);

  float k = 1.0f / in_channels;
  for (int i = 0; i < out_channels * in_channels; i++) {
    weight[i] = randomFloat(-k, k);
  }
  for (int i = 0; i < out_channels; i++) {
    bias[i] = randomFloat(-k, k);
  }
}

Linear::~Linear() {
  free(output);
  free(bias);
  free(weight);
  free(error);
  free(u_bias);
  free(u_weight);
  free(gradient);
}

void Linear::forward(float* input) {
  for (int i = 0; i < out_channels; i++) {
    float x = bias[i];
    for (int j = 0; j < in_channels; j++) {
      x += input[j] * weight[i*in_channels+j];
    }
    output[i] = x;
  }
   
  //softmax
  float max = output[0];
  for (int i = 1; i < out_channels; i++) {
    if (max < output[i]) {
      max = output[i];
    }
  }
  float sum = 0.0001;
  for (int i = 0; i < out_channels; i++) {
    output[i] = exp(output[i] - max);
    sum += output[i];
  }
  for (int i = 0; i < out_channels; i++) {
    output[i] /= sum;
  }
}

void Linear::backward(float* input, float* src_error) {
  for (int i = 0; i < in_channels; i++) {
    src_error[i] = 0;
  }

  for (int i = 0; i < out_channels; i++) {
    gradient[i] = output[i] + error[i];
    for (int j = 0; j < in_channels; j++) {
      src_error[j] += weight[i*in_channels+j] * gradient[i];
      u_weight[i*in_channels+j] += gradient[i] * input[j];
    }
    u_bias[i] += gradient[i];
  }
}

void Linear::update(float rate) {
  for (int i = 0; i < out_channels; i++) {
    bias[i] -= rate * u_bias[i];
    error[i] = 0;
    u_bias[i] = 0;
    for (int j = 0; j < in_channels; j++) {
      weight[i*in_channels+j] -= rate * u_weight[i*in_channels+j];
      u_weight[i*in_channels+j] = 0;
    }
  }
}
