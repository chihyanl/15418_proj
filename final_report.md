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
In order to parallelize and analyze the effectiveness of layer fusion on GPUs, we implemented a simple CNN targeted for MNIST. To have the best control over the details and ensure that our comparisons are on the same basis, we implemented our designs from scratch with C++ and CUDA. Our implementations target the GHC machines with the NVIDIA GeForce RTX 2080 GPU.

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

#### CPU+CUDA CNN Implementation
To ease the development process, we started from a CPU implementation (inspired by [2]) and targeted for a higher than 97% correctness on the MNIST. Our final CNN implmentation is structured as shown in Table 1, achieving near 98% correctness with 2 epochs. Porting the CPU implementation to GPU, we unrolled the outer 3 loops in the forward/backward passes of the convolutional layers and flattened the fully connected layer. With the CUDA implementation, we were able to shorten the training time from 207.13 seconds to 49.57 seconds and the testing time from 8.804 seconds to 0.703 seconds. The CUDA implementation has each thread compute one output for each of the layers, allowing a highly parallelized approach without dependencies.

#### Layer Fusion Implementation
With the conventional CNN implementation in CUDA, we implemented layer fusion in tiles, where each thread block would compute the fused layer's output in the tile. By tiling, we are able to localize the computation, allowing the intermediate data to be stored locally in the shared memory. Due to the shared memory not being retained between kernels, our implementation uses the recomputation approach, where the overlapping regions are recomputed instead of reused. In contrast to the FPGA implmentation in [1], the reuse approach would require the overlapped regions' data to be stored in the global memory, defeating the goal of layer fusion.

The fused layer forward passes are structured as shown below:
```
copy input data from global memory to shared memory buffer
compute 1st layer forward pass
compute 2nd layer forward pass
...
compute nth layer forward pass
copy nth layer output from shared memory buffer to global memory
```
##### Thread Mapping
The number of threads per block is determined by the maximum of each layer's product of output channel size and the tile size: $\max\limits_{i}(out\\_channel_i \times tile\\_width_i \times tile\\_height_i\)$. Additionally, we attempted using 256 threads per block, where each thread is statically assigned with a set of channel and tile coordinate combinations, but it turns out to perform worse potentially due to the limited parallelization and worse load balance. In the final implementation, we implemented the former approach, which also limits the maximum output channel and tile size product to 1024 as the maximum of threads per shared memory is 1024.
##### Block Mapping
The number of thread blocks is determined by the last layer's output dimension and the tile size: $\frac{width\\_out+tile\\_width-1}{tile\\_width}\frac{height\\_out+tile\\_height-1}{tile\\_height}$. This ensures coverage for the entire output while each thread block performs the computation for each tile on the shared memory.

### Results
We are successful in reaching our goals:
* we implemented a baseline CNN for MNIST using conventional layer-by-layer computation
* we implmented layer fusion on CUDA with 2 layers fused, 3 layers fused, and 4 layers fused
* our implmentation iterates over the different approaches
* we constructed a simple Python script for testing with multiple trials
* we analyzed and studied the effect of layer fusion on the GPU

#### Optimization
While we attempted to optimize both the original CUDA implmentation and the layer fusion implementation, the primary goal is to study the effect of layer fusion on GPUs. In order to ensure that we have the same basis for comparison, the approaches are developed from the same base code.

#### Experiment Setup
Our experiment is done with the handwritten digits MNIST with a 28x28x1 input, 60000 training samples, and 10000 testing samples. The training is done with 2 epochs, then the trained weights/biases are tested with the different implementations: the conventional layer-by-layer on the CPU, the conventional layer-by-layer on the GPU, first 2 layers fused on the GPU, first 3 layers fused on the GPU, first 4 layers fused, and 2x2 layers fused (where the first 2 layers are fused and the 3rd/4th layers are fused). The primary interest is the performance measured with the wall-clock time of each implementation. The secondary interest is the shared memory and threads per block utilization and their correlation with performance.

### Resources
> [1] M. Alwani, H. Chen, M. Ferdman and P. Milder, "Fused-layer CNN accelerators," 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), Taipei, Taiwan, 2016, pp. 1-12, doi: [10.1109/MICRO.2016.7783725.](https://doi.org/10.1109/MICRO.2016.7783725)
>
>  [2] Euske, “Convolutional Neural Network in C (for educational purposes),” GitHub, https://github.com/euske/nn1/tree/master (accessed Dec. 9, 2024). 

### Contribution
| Member | Work | Distribution |
|-|-|-|
| Lawrence Lo | CPU+CUDA Implementation, Analysis, Report | 50% |
| Eshita Shrawan | Layer Fusion Implmentation, Analysis, Report | 50% |
