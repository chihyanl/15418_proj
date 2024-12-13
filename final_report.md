---
layout: page
title: Final Report
permalink: /final_report/
usemathjax: true
---
## Title
CNN with Layer Fusion on NVIDIA GPU
## Repository URL
https://github.com/chihyanl/15418_proj/
## Table of Content
1. [Summary](#summary)
2. [Background](#background)
3. [Approach](#approach)
   1. [CPU+CUDA CNN Implementation](#cpu_cuda_cnn_implementation)
   2. [Layer Fusion Implementation](#layer_fusion_implementation)
       1. [Thread Mapping](#thread_mapping)
       2. [Block Mapping](#block_mapping)
4. [Results](#results)
   1. [Optimization](#optimization)
   2. [Experiment Setup](#experiment_setup)
   3. [Overall Performance](#overall_performance)
   4. [Impact of Tile Size on Performance](#tile_size_performance)
   5. [Impact of Network Size on Speedup](#network_size_speedup)
      1. [Altering L1's Output Dimension](#l1_output_dimension)
      2. [Altering L1's Output Channel Size](#l1_output_channel)
   6. [Conclusion](#conclusion)
5. [Resources](#resources)
6. [Contribution](#contribution)
7. [Data](#data)

## Summary <a name="summary"></a>
We are going to implement CNN with layer fusion on NVIDIA GPU. The project's goal is to port the original FPGA implementation to an NVIDIA GPU implementation.
## Background <a name="background"></a>
Conventional convolution layer has results stored locally and written to the memory at the end of each tile. This requires many memory inferences, which are limited by the memory bandwidth and hinders scalability. Layer fusion attempts to reduce the number of memory inferences by storing tiles of intermediate data to higher-level storage units, instead of inferencing the global memory. With layer fusion, the memory bandwidth requirement and scalability issue can be alleviated.

Layer fusion uses tiles of blocked kernels to form a pyramid. By using the outputs of the first few layers immediately reduces the need to store/read them from the memory. However, with the pyramid structure, there are overlapping intermediate data used by different tiles. These intermediate data can either be recomputed or reused, each having their own trade-off. Recomputing is a simple approach but requires more operations, whereas reusing avoids the extra computation but requires more local storage.

<p align="center">
  <img src="../fusion_pyramid.png" width=400> <br>
  Fig 1. Example of a single pyramid and a multi-pyramid applied over four layers [1]
</p>

## Approach <a name="approach"></a>
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

### CPU+CUDA CNN Implementation <a name="cpu_cuda_cnn_implementation"></a>
To ease the development process, we started from a CPU implementation (inspired by [2]) and targeted for a higher than 97% correctness on the MNIST. Our final CNN implementation is structured as shown in Table 1, achieving near 98% correctness with 2 epochs. Porting the CPU implementation to GPU, we unrolled the outer 3 loops in the forward/backward passes of the convolutional layers and flattened the fully connected layer. With the CUDA implementation, we were able to shorten the training time from 207.13 seconds to 49.57 seconds and the testing time from 8.804 seconds to 0.703 seconds. The CUDA implementation has each thread compute one output for each of the layers, allowing a highly parallelized approach without dependencies.

### Layer Fusion Implementation <a name="layer_fusion_implementation"></a>
With the conventional CNN implementation in CUDA, we implemented layer fusion in tiles, where each thread block would compute the fused layer's output in the tile. By tiling, we are able to localize the computation, allowing the intermediate data to be stored locally in the shared memory. Due to the shared memory not being retained between kernels, our implementation uses the recomputation approach, where the overlapping regions are recomputed instead of reused. In contrast to the FPGA implementation in [1], the reuse approach would require the overlapped regions' data to be communicated via the global memory, defeating the goal of layer fusion.

The fused layer forward passes are structured as shown below:
```
copy input data from global memory to shared memory buffer
compute 1st layer forward pass
compute 2nd layer forward pass
...
compute nth layer forward pass
copy nth layer output from shared memory buffer to global memory
```
#### Thread Mapping <a name="thread_mapping"></a>
The number of threads per block is determined by the maximum of each layer's product of output channel size and the tile size:

<div align="center"><img src="../figure/threadmap_equation.png" width=400></div>

Additionally, we attempted using 256 threads per block, where each thread is statically assigned with a set of channel and tile coordinate combinations, but it turns out to perform worse potentially due to the limited parallelization and worse load balance. In the final implementation, we implemented the former approach, which also limits the maximum output channel and tile size product to 1024 as the maximum of threads per shared memory is 1024.
#### Block Mapping <a name="block_mapping"></a>
The number of thread blocks is determined by the *n<sup>th</sup>* layer's output dimension and the tile size:

<div align="center"><img src="../figure/blockmap_equation.png" width=400></div>

This ensures coverage for the entire output while each thread block performs the computation for each tile on the shared memory.

## Results <a name="results"></a>
We are successful in reaching our goals:
* we implemented a baseline CNN for MNIST using conventional layer-by-layer computation
* we implemented layer fusion on CUDA with 2 layers fused, 3 layers fused, and 4 layers fused
* our implementation iterates over the different approaches
* we constructed a simple Python script for testing with multiple trials
* we analyzed and studied the effect of layer fusion on the GPU

### Optimization <a name="optimization"></a>
While we attempted to optimize both the original CUDA implementation and the layer fusion implementation, the primary goal is to study the effect of layer fusion on GPUs. In order to ensure that we have the same basis for comparison, the approaches are developed from the same base code.

### Experiment Setup <a name="experiment_setup"></a>
Our experiment is done with the handwritten digits MNIST with a 28x28x1 input, 60000 training samples, and 10000 testing samples. The training is done with 2 epochs, then the trained weights/biases are tested with the different implementations: the conventional layer-by-layer on the CPU, the conventional layer-by-layer on the GPU, first 2 layers fused on the GPU, first 3 layers fused on the GPU, first 4 layers fused, and 2x2 layers fused (where the first 2 layers are fused and the 3rd/4th layers are fused). The primary interest is the performance measured with the wall-clock time of the testing phase of each implementation. The secondary interest is the shared memory and threads per block utilization and their correlation with performance.

### Overall Performance <a name="overall_performance"></a>
<p align="center">
  <img src="../figure/total_runtime_vs_implementation.png" width=800> <br>
  Fig 2. Total Runtime vs Implementation
</p>
Figure 2 shows the best timing of each implementation. 'Result Time' denotes the time taken to generate the prediction, while 'L1 Time', 'L2 Time', 'L3 Time', 'L4 Time', and 'L5 Time' represent the time taken for their respective layer, as indicated in Table 1.

In general, the performance improves as the number of fused layers increases, with the exception of when 4 layers are fused. The improvements result from reduced communication with global memory. Specifically, fusing 2 layers eliminates 1 set of input/output communications, fusing 3 layers eliminates 2 sets of input/output communications, and fusing 2x2 layers eliminates 2 sets of input/output communications.

While fusing 3 layers and fusing 2x2 layers both eliminate 2 sets of input/output communications, fusing 3 layers requires 151 recomputations per thread block whereas 2x2 layers only requires 64 recomputations per thread block. The increasing recomputation requires more threads per block to be allocated, using up resources that could otherwise be used for 'useful' work. Additionally, fusing 3 layers utilizes 864 threads, which is a high thread count that leads to lower utilization of the streaming multiprocessors. In contrast, fusing 2x2 layers utilizes significantly fewer threads (216 and 128 threads respectively), achieving a better utilization and performance.

Fusing 4 layers has the similar issues stated with fusing 3 layers but more severe. The increased number of fused layers requires 175 recomputations per thread block, limiting the tile size to 1x1 to stay within the 1024 threads per block limit. This results in 121 thread blocks (approximately 4.8 times more thread blocks), which degrades performance. Additionally, fusing four layers utilizes 864 threads, which leads to reduced utilization and subsequently lower performance.

### Impact of Tile Size on Performance <a name="tile_size_performance"></a>
<p align="center">
   <img src="../figure/2_layer_fused_tile_size.png" width=800> <br>
   Fig 3. 2 Layer Fused Tile Size Results <br>
</p>
<p align="center">
   <img src="../figure/3_layer_fused_tile_size.png" width=800> <br>
   Fig 4. 3 Layer Fused Tile Size Results
</p>

As shown in Figure 3 and Figure 4, a smaller tile size results in less threads per block and shared memory utilization, since the maximum tile size of the layers is smaller and less intermediate data is produced. However, the best performing tile size is determined by the threads per block, number of blocks, and recomputations per thread block combination.

For example, when fusing 2 layers, a 5x5 tile size requires 864 threads per block, 9 thread blocks, and 119 recomputations per thread per output channel, whereas a 2x2 tile size requires 216 threads per block, 49 thread blocks, and 32 recomputations per thread per output channel. The 2x2 tile excels in performance with all 3 factors, resulting in an improved performance.

Additionally, when fusing 3 layers, a 3x3 tile size requires 864 threads per block, 25 thread blocks, and 151 recomputations per thread per output channel, whereas a 1x1 tile size requires 384 threads per block, 169 thread blocks, and 71 recomputations per thread per output channel. The 1x1 tile size has lower recomputations per thread, which improves performance, but the number of threads per block and thread blocks combination results in significantly worse utilization, hindering the performance.

All three factors—the number of threads per block, the number of thread blocks, and the number of recomputations per thread—are affected by the tile size. Finding the optimal tile size to balance these factors is challenging, especially as the number of fused layers increases. Generally, fusing only two layers at a time is ideal for achieving this balance, as it requires fewer threads per block, increases the number of thread blocks for better load balancing (given an ideal number of threads per block), and reduces recomputations per thread. This results in better performance and utilization for 'useful' work as shown with the 2x2 fused layers.

### Impact of Network Size on Speedup <a name="network_size_speedup"></a>
To examine how network size influences speedup, we designed two experiments. In the first experiment, we modified the output dimension of L1, excluding the output channel, and adjusted the corresponding values of the remaining layers. In the second experiment, we altered L1's output channel size and matched L2's input channel accordingly. In our experiment, we compare the speedup of the conventional CUDA CNN with the 2x2 layer fusion implementation, using a tile size of 2x2 for both fused stacks. This specific structure was chosen because it demonstrated the best performance in our prior tests.

#### Altering L1’s Output Dimension <a name="l1_output_dimension"></a>
<p align="center">
   <img src="../figure/l1_dim_speedup.png" width=800> <br>
   Fig 5. Speedup by Altering L1 Output Dimension
</p>

Figure 5 illustrates the overall speedup and the speedup of the fused convolutional layers (L1 to L4) for various network sizes, achieved by modifying L1's output dimension. The network size is represented as the total of the input and output sizes of the convolutional layers.
As shown in Figure 5, when the network size increases, there is a decrease in speedup. The reduction in speedup can be attributed to the constant number of threads per thread block and recomputations per thread block, combined with an increase in the number of thread blocks. This results in lower utilization and more recomputation. In the case of fusing 2x2 layers (with 2x2 and 2x2 tile sizes), load balancing is not a primary concern since the setup already has a sufficient number of thread blocks. However, as the network size increases, the rise in thread blocks exacerbates the issues of utilization and recomputation, resulting in a decrease in speedup.

Additionally, having a network size that is too small may lead to diminishing returns as shown with the L1+L2+L3+L4 speedup. This can be due to the lack of parallelism with the smaller networks. When the number of thread blocks falls below the optimal level, it limits parallelism and leads to diminishing returns on speedup. Furthermore, we do not observe the same effect in the overall speedup because, in a smaller network, a larger fraction of the program is attributed to the convolutional layers. According to Amdahl’s Law, the speedup increases as the optimizable fraction increases.

#### Altering L1’s Output Channel Size <a name="l1_output_channel"></a>
<p align="center">
   <img src="../figure/l1_channel_speedup.png" width=800> <br>
   Fig 6. Speedup by Altering L1 Output Channel Size
</p>

Figure 6 illustrates the overall speedup and the speedup of the fused convolutional layers (L1 to L4) for various network sizes, achieved by modifying L1's output channel size. The network size is represented as the size of L1’s output channel.

Similar to altering L1's output dimension, a decrease in network size also leads to an increase in speedup, albeit a minor one. The minor differences in speedup can be attributed to the slight change in network size, as we are only modifying one of the relatively small layers. As the number of output channels increases, the number of threads per thread block while the recombinations per thread block and number of thread blocks remain constant. With less threads per thread block, there is higher utilization as more threads are allocated to do ‘useful’ work.

As depicted in Figure 6, the overall trend indicates an increase in speedup with smaller network sizes. However, there are notable spikes at output channel sizes of 9, 11, and 13. This is attributed to a better thread and block mapping that results in better utilization. For instance, with an output channel size of 9, there are 324 threads per thread block. In contrast, an output channel size of 8 results in 288 threads per thread block, leading to more threads being ‘unproductive’. Furthermore, there's a notable drop in speedup at an output channel size of 15. This drop is due to ineffective thread and block mapping, where 540 threads per thread block are used, resulting in a significant number of 'unproductive' threads.

### Conclusion <a name="conclusion"></a>
Layer fusion is effective in improving the performance of a CNN by reducing the number of global memory transfers. While having more fused layers improves performance, it is better to fuse multiple fewer-layer stacks instead of fusing a single large multilayer stack due to the lower utilization and higher recomputations with the single multilayer stack. Additionally, due to the limitations on GPU thread and block mapping for layer fusion, it is advisable to keep the layer sizes small to ensure effective utilization of resources.

While we cannot conclude whether a GPU implementation is better or worse than the original FPGA implementation in [1] due to the lack of comparison against an FPGA, we prove that the GPU implementation could serve as an alternative machine for the task. Despite having the need to recompute due to the limitation with kernel memory retention and communication between thread blocks, the GPU implementation is able to provide gain in performance that speeds up the inference time of a CNN.

## Resources <a name="resources"></a>
> [1] M. Alwani, H. Chen, M. Ferdman and P. Milder, "Fused-layer CNN accelerators," 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), Taipei, Taiwan, 2016, pp. 1-12, doi: [10.1109/MICRO.2016.7783725.](https://doi.org/10.1109/MICRO.2016.7783725)
>
>  [2] Euske, “Convolutional Neural Network in C (for educational purposes),” GitHub, https://github.com/euske/nn1/tree/master (accessed Dec. 9, 2024). 

## Contribution <a name="contribution"></a>
| Member | Work | Distribution |
|-|-|-|
| Lawrence Lo | CPU+CUDA Implementation, Analysis, Report | 50% |
| Eshita Shrawan | Layer Fusion Implmentation, Analysis, Report | 50% |

## Data <a name="data"></a>

<div align="center"> <b>
  Table 2 <br>
  CPU and CUDA Conventional CNN
</b>
   
|  | Accuracy | Total Time | L1 Time | L2 Time | L3 Time | L4 Time | L5 Time | Result Time | 
|-|-|-|-|-|-|-|-|-|
| CPU | 97.01% | 8.80399 | 1.57252 | 3.70156 | 2.94060 | 0.54988 | 0.03877 | 0.00012 | 
| Cuda | 97.49% | 0.70306 | 0.08659 | 0.12945 | 0.18625 | 0.13230 | 0.15496 | 0.01291 |
</div>

<div align="center"> <b>
  Table 3 <br>
  2 Layers Fused
</b>
   
| Tile Size | Threads | Buffer | Accuracy | Total Time | L1-L2 Time | L3 Time | L4 Time | L5 Time | Result Time |
|-|-|-|-|-|-|-|-|-|-|
| 5x5 | 864 | 1520 | 97.49% | 0.66525 | 0.17605 | 0.18708 | 0.13280 | 0.15552 | 0.01320 | 
| 4x4 | 600 | 1052 | 97.49% | 0.64292 | 0.15923 | 0.18567 | 0.13141 | 0.15417 | 0.01201 | 
| 3x3 | 384 | 672 | 97.49% | 0.64270 | 0.15846 | 0.18580 | 0.13155 | 0.15432 | 0.01212 | 
| 2x2 | 216 | 380 | 97.49% | 0.65137 | 0.15841 | 0.18798 | 0.13375 | 0.15648 | 0.01412 | 
| 1x1 | 96 | 176 | 97.49% | 0.68353 | 0.19847 | 0.18605 | 0.13173 | 0.15447 | 0.01239 | 
</div>

<div align="center"> <b>
  Table 4 <br>
  3 Layers Fused
</b>
   
| Tile Size | Threads | Buffer | Accuracy | Total Time | L1-L3 Time | L4 Time | L5 Time | Result Time | 
|-|-|-|-|-|-|-|-|-|
| 3x3 | 864 | 1592 | 97.39% | 0.60997 | 0.30917 | 0.13312 | 0.15486 | 0.01247 | 
| 2x2 | 600 | 1084 | 97.39% | 0.74178 | 0.43985 | 0.13368 | 0.15531 | 0.01258 | 
| 1x1 | 384 | 680 | 97.39% | 0.80323 | 0.50138 | 0.13359 | 0.15532 | 0.01258 | 
</div>

<div align="center"> <b>
  Table 5 <br>
  4 Layers Fused
</b>
   
| Tile Size | Threads | Buffer | Accuracy | Total Time | L1-L4 Time | L5 Time | Result Time |
|-|-|-|-|-|-|-|-|
| 1x1 | 864 | 1596 | 97.39% | 1.05306 | 0.88485 | 0.15524 | 0.01273 | 
</div>

<div align="center"> <b>
  Table 6 <br>
  2x2 Layers Fused
</b>
   
| Tile Sizes | Threads1 | Buffer1 | Threads2 | Buffer2 | Accuracy | Total Time | L1-L2 Time | L3-L4 Time | L5 Time | Result Time |
|-|-|-|-|-|-|-|-|-|-|-|
| 5x5, 5x5 | 864 | 1520 | 392 | 1788 | 97.49% | 0.58475 | 0.17451 | 0.24294 | 0.15462 | 0.01239 | 
| 5x5, 4x4 | 864 | 1520 | 288 | 1376 | 97.49% | 0.58167 | 0.17524 | 0.23817 | 0.15534 | 0.01262 | 
| 5x5, 3x3 | 864 | 1520 | 200 | 1020 | 97.49% | 0.58056 | 0.17507 | 0.23737 | 0.15518 | 0.01265 | 
| 5x5, 2x2 | 864 | 1520 | 128 | 720 | 97.49% | 0.57410 | 0.17480 | 0.23158 | 0.15500 | 0.01243 | 
| 5x5, 1x1 | 864 | 1520 | 72 | 476 | 97.49% | 0.58985 | 0.17416 | 0.24836 | 0.15467 | 0.01237 | 
| 4x4, 5x5 | 600 | 1052 | 392 | 1788 | 97.49% | 0.56827 | 0.15855 | 0.24270 | 0.15445 | 0.01227 | 
| 4x4, 4x4 | 600 | 1052 | 288 | 1376 | 97.49% | 0.56335 | 0.15879 | 0.23731 | 0.15455 | 0.01241 | 
| 4x4, 3x3 | 600 | 1052 | 200 | 1020 | 97.49% | 0.56308 | 0.15884 | 0.23678 | 0.15471 | 0.01246 | 
| 4x4, 2x2 | 600 | 1052 | 128 | 720 | 97.49% | 0.55898 | 0.15919 | 0.23173 | 0.15516 | 0.01260 | 
| 4x4, 1x1 | 600 | 1052 | 72 | 476 | 97.49% | 0.57553 | 0.15893 | 0.24869 | 0.15519 | 0.01242 | 
| 3x3, 5x5 | 384 | 672 | 392 | 1788 | 97.49% | 0.56659 | 0.15695 | 0.24264 | 0.15444 | 0.01226 | 
| 3x3, 4x4 | 384 | 672 | 288 | 1376 | 97.49% | 0.56435 | 0.15800 | 0.23801 | 0.15544 | 0.01260 | 
| 3x3, 3x3 | 384 | 672 | 200 | 1020 | 97.49% | 0.56220 | 0.15750 | 0.23694 | 0.15497 | 0.01251 | 
| 3x3, 2x2 | 384 | 672 | 128 | 720 | 97.49% | 0.55543 | 0.15718 | 0.23101 | 0.15447 | 0.01247 | 
| 3x3, 1x1 | 384 | 672 | 72 | 476 | 97.49% | 0.57224 | 0.15679 | 0.24820 | 0.15455 | 0.01240 | 
| 2x2, 5x5 | 216 | 380 | 392 | 1788 | 97.49% | 0.56556 | 0.15584 | 0.24265 | 0.15445 | 0.01232 | 
| 2x2, 4x4 | 216 | 380 | 288 | 1376 | 97.49% | 0.56058 | 0.15597 | 0.23717 | 0.15463 | 0.01251 | 
| 2x2, 3x3 | 216 | 380 | 200 | 1020 | 97.49% | 0.55976 | 0.15591 | 0.23648 | 0.15457 | 0.01251 | 
| 2x2, 2x2 | 216 | 380 | 128 | 720 | 97.49% | 0.55396 | 0.15578 | 0.23094 | 0.15448 | 0.01247 | 
| 2x2, 1x1 | 216 | 380 | 72 | 476 | 97.49% | 0.57092 | 0.15555 | 0.24814 | 0.15453 | 0.01240 | 
| 1x1, 5x5 | 96 | 176 | 392 | 1788 | 97.49% | 0.60747 | 0.19748 | 0.24267 | 0.15456 | 0.01247 | 
| 1x1, 4x4 | 96 | 176 | 288 | 1376 | 97.49% | 0.60193 | 0.19748 | 0.23712 | 0.15457 | 0.01247 | 
| 1x1, 3x3 | 96 | 176 | 200 | 1020 | 97.49% | 0.60117 | 0.19749 | 0.23636 | 0.15458 | 0.01246 | 
| 1x1, 2x2 | 96 | 176 | 128 | 720 | 97.49% | 0.59548 | 0.19726 | 0.23094 | 0.15455 | 0.01245 | 
| 1x1, 1x1 | 96 | 176 | 72 | 476 | 97.49% | 0.61256 | 0.19701 | 0.24814 | 0.15468 | 0.01243 | 
</div>

<div align="center"> <b>
  Table 7 <br>
  Altering L1 Output Dimension
</b>
   
| Network Size | Cuda Total | Cuda (L1-L4) | 2x2 Layer Total | 2x2 Layer (L1-L4) | Total Speedup | Speedup (L1-L4) |
|-|-|-|-|-|-|-|
| 22604 | 1.00455 | 0.56584 | 0.89325 | 0.45489 | 1.12460 | 1.24390 | 
| 19300 | 0.94660 | 0.56517 | 0.82087 | 0.43949 | 1.15317 | 1.28596 | 
| 16268 | 0.89935 | 0.57136 | 0.75694 | 0.42893 | 1.18814 | 1.33207 | 
| 13508 | 0.84402 | 0.56669 | 0.69055 | 0.41298 | 1.22225 | 1.37219 | 
| 11020 | 0.78890 | 0.55645 | 0.63319 | 0.40034 | 1.24591 | 1.38997 | 
| 8804 | 0.62786 | 0.45824 | 0.50337 | 0.33349 | 1.24731 | 1.37409 | 
| 6860 | 0.69546 | 0.53056 | 0.54958 | 0.38504 | 1.26545 | 1.37794 | 
| 3828 | 0.66637 | 0.52749 | 0.51709 | 0.37870 | 1.28869 | 1.39291 | 
| 2892 | 0.63816 | 0.52415 | 0.49165 | 0.37829 | 1.29801 | 1.38557 | 
</div>

<div align="center"> <b>
  Table 8 <br>
  Altering L1 Output Channel Size
</b>
   
| Output Channel Size | Cuda Total | Cuda (L1-L4) | 2x2 Layer Total | 2x2 Layer (L1-L4) | Total Speedup | Speedup (L1-L4) |
|-|-|-|-|-|-|-|
| 16 | 0.81435 | 0.64842 | 0.77480 | 0.60957 | 1.05105 | 1.06373 | 
| 15 | 0.79759 | 0.63278 | 0.75291 | 0.58842 | 1.05935 | 1.07539 | 
| 14 | 0.78770 | 0.62202 | 0.63575 | 0.47024 | 1.23902 | 1.32277 | 
| 13 | 0.78034 | 0.61314 | 0.62315 | 0.45663 | 1.25227 | 1.34276 | 
| 12 | 0.76797 | 0.60113 | 0.61862 | 0.45210 | 1.24142 | 1.32964 | 
| 11 | 0.75365 | 0.58809 | 0.60006 | 0.43506 | 1.25596 | 1.35176 | 
| 10 | 0.74063 | 0.57473 | 0.59682 | 0.43119 | 1.24095 | 1.33291 | 
| 9 | 0.73126 | 0.56488 | 0.57895 | 0.41364 | 1.26308 | 1.36563 | 
| 8 | 0.71614 | 0.55155 | 0.57137 | 0.40673 | 1.25336 | 1.35606 | 
| 7 | 0.70379 | 0.53921 | 0.55948 | 0.39439 | 1.25792 | 1.36722 | 
| 6 | 0.69560 | 0.52952 | 0.54984 | 0.38460 | 1.26509 | 1.37683 | 
| 5 | 0.68522 | 0.51973 | 0.54033 | 0.37476 | 1.26816 | 1.38686 | 
| 4 | 0.67385 | 0.50788 | 0.53286 | 0.36737 | 1.26460 | 1.38247 | 
</div>
