---
layout: page
title: Proposal
permalink: /
---
### Title
CNN with Layer Fusion on NVIDIA GPU
### Team
* Lawrence Lo
* Eshita Shrawan

### URL
<https://chihyanl.github.io/15418_proj/>
### Summary
We are going to implement CNN with layer fusion on NVIDIA GPU. The project's goal is to port the original FPGA implementation to an NVIDIA GPU implementation.
### Background
Conventional convolution layer has results stored locally and written to the memory at the end of each tile. This requires many memory inferences, which are limited by the memory bandwidth and hinders scalability. Layer fusion attempts to reduce the number of memory inferences by storing tiles of intermediate data to higher-level storage units, instead of inferencing the global memory. With layer fusion, the memory bandwidth requirement and scalability issue can be alleviated.

Layer fusion uses tiles of blocked kernels to form a pyramid. By using the outputs of the first few layers immediately reduces the need to store/read them from the memory. However, with the pyramid structure, there are overlapping intermediate data used by different tiles. These intermediate data can either be recomputed or reused, each having their own trade-off. Recomputing is a simple approach but requires more operations, whereas reusing avoids the extra computation but requires more local storage.

<p align="center">
  <img src="./fusion_pyramid.png" width=400>
</p>

### The Challenge
As outlined in the paper, managing the data dependency between the kernel threads during implementation is the biggest challenge. The paper mentions that *current GPU programming abstractions make it challenging to precisely orchestrate the thread behavior and buffer management of layer fusiocurrent GPU programming abstractions make it challenging to precisely orchestrate the thread behavior and buffer management of layer fusion*.

To properly benefit from layer fusion, we have to implement kernel function(s) that can properly cache intermediate date between layers and synchronize computation between layers. Computation in the later layers have dependency on earlier layers' result, while also requiring less computation in a pyramid fashion as shown in the diagram of the previous section. We also need to ensure all cached data fits in the GPU's memory. Ensuring all of these constraints while maintaining high performance poses a huge challenge.

### Resources
Lorem Ipsum

### Goals and Deliverables
#### Minimum goal
Over the next month, we plan to achieve the following
* Implementation of baseline AlexNet using layer-by-layer computation
* Implementation of AlexNet with first two layers fused
* Implementation of testing framework for benchmarking
* Performance analysis of both implementations

Given the short amount of time for implementation, our realistic target is to implement AlexNet from scratch both running on GPU, but one computing layer-by-layer and another with fused layer. We will reduce the complexity of the problem by fixing the network structure to AlexNet (the same neural network used in the cited paper), and fixing the fusion layer only to the first two.

During the poster session, we will be demonstrating the speedup graphs and live demo of both implementations. We are hoping to learn about the tradeoffs between speedup and keeping more data in the GPU. We also wish to compare this result with the FPGA implementation outlined in the paper.

#### Stretch goal
If we have enough time, we will work on the following goals:
* Implementation of general solution for layer fusion in GPU
* Running an image classification model on our layer fusion implementation

The second goal especially will make a flashy demo for the poster session.

### Platform Choice
We will be implementing this project in C++ and CUDA. We will be using the GPUs in the GHC cluster equipped with NVIDIA GeForce RTX 2080. Our project specifically explores CNN Layer Fusion in GPU, and this is the most accessible GPUs we can utilize.

### Schedule
Lorem Ipsum

