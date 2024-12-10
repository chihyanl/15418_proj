---
layout: page
title: Final Report
permalink: /final_report/
---
### Title
CNN with Layer Fusion on NVIDIA GPU

### Summary
We are going to implement CNN with layer fusion on NVIDIA GPU. The project's goal is to port the original FPGA implementation to an NVIDIA GPU implementation.
### Background
Conventional convolution layer has results stored locally and written to the memory at the end of each tile. This requires many memory inferences, which are limited by the memory bandwidth and hinders scalability. Layer fusion attempts to reduce the number of memory inferences by storing tiles of intermediate data to higher-level storage units, instead of inferencing the global memory. With layer fusion, the memory bandwidth requirement and scalability issue can be alleviated.

Layer fusion uses tiles of blocked kernels to form a pyramid. By using the outputs of the first few layers immediately reduces the need to store/read them from the memory. However, with the pyramid structure, there are overlapping intermediate data used by different tiles. These intermediate data can either be recomputed or reused, each having their own trade-off. Recomputing is a simple approach but requires more operations, whereas reusing avoids the extra computation but requires more local storage.

<p align="center">
  <img src="../fusion_pyramid.png" width=400>
</p>
<p align="center">
  Fig 1. Example of a single pyramid and a multi-pyramid applied over four layers [1]
</p>

### Approach
In order to parallelize and analyze the effectiveness of layer fusion on GPUs, we implemented a simple CNN targeted for MNIST. To have the best control over the details and ensure that our comparisons are on the same basis, we implemented our designs from scratch with C++ and CUDA.

<div align="center"> <b>
  Table 1 <br>
  CNN Configuration
</b>
  
| Layer | Type | Output Size | Kernel Size | Stride | Pad | Number of Weights | Number of Biases |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| L0 | Input | 28x28x1 | - | - | - | - | - |
| L1 | Convolution | 28x28x6 | 5x5 | 1 | 2 | 150 | 6 |
| L2 | Convolution | 13x13x16 | 4x4 | 2 | 0 | 1536 | 16 |
| L3 | Convolution | 13x13x8 | 3x3 | 1 | 1 | 1152 | 8 |
| L4 | Convolution | 11x11x4 | 3x3 | 1 | 0 | 288 | 4 |
| L5 | Fully Connected | 10 | - | - | - | 4840 | 10 |
</div>

To ease the development process, we started from a CPU implementation and targeted for a higher than 97% correctness on the MNIST. Our final CNN implmentation is structured as shown in Table 1, achieving near 98% correctness with 2 epochs.

### Results

### Resources
> [1] M. Alwani, H. Chen, M. Ferdman and P. Milder, "Fused-layer CNN accelerators," 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), Taipei, Taiwan, 2016, pp. 1-12, doi: [10.1109/MICRO.2016.7783725.](https://doi.org/10.1109/MICRO.2016.7783725)

### Contribution
| Member | Work | Distribution |
|-|-|-|
| Lawrence Lo | CPU+CUDA Implementation, Analysis, Report | 50% |
| Eshita Shrawan | Layer Fusion Implmentation, Analysis, Report | 50% |
