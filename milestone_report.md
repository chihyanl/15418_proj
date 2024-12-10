---
layout: page
title: Milestone Report
permalink: /milestone_report/
---
## Title
CNN with Layer Fusion on NVIDIA GPU
## Team
* Lawrence Lo
* Eshita Shrawan

## Work completed so far
So far, we've implemented CNN for MNIST in both CPU version and the CUDA version, along with the testing framework for validation. The model architecture we're using contains 4 convolutional layers (28x28x6 -> 13x13x16 -> 13x13x8 -> 11x11x4) and one fully connected layer (484->10). We are training on 60k images, and testing it on 10k images. Due to the simple nature of the problem, we're able to achieve ~98% accuracy with only 2 epochs.

We are working on layer fusion implementation as of now, and expect to finish baseline implementation by the middle of this week. We are slightly behind the original schedule as we did not properly account for the Thanksgiving break, but we are on track to finish our initial goals by the deadline. We anticipate that we can implement a flashy demo with visualizers amongst our stretch goals for the poster session.

## Poster Session Plan

For the poster session, we will present our timing results and include memory metrics (how much memory was used in the GPU) graphically - this will be crucial in demonstrating the tradeoffs between using the layer fusion or not. For the demo, we will classify images using both models and compare the speed.

## Preliminary Results

Following is the performance metric of both versions in the format of total time (First two Conv Layer Forward time):
| CPU Training time | CPU Inference time | GPU Training time | GPU Inference time | 
|-|-|-|-|
| 207.13s (63.33s) | 8.801s (5.275s) | 33.12s (0.35s) | 0.383s (0.105s) |

## Concerns
We have no major concerns as of now. Work that is left is simply a matter of coding and anylsis.

## Schedule
Below is the updated schedule for the remainder of the class:

| | |
|-|-|
| Week 4 H1 | Finish Implementing Layer Fusion |
| Week 4 H2 | Layer Fusion optmization & Analysis |
| Week 5 H1 | Work on demo stretch goals & Work on Final Report & Poster |
| Week 5 H2 | Polish Final Report & Poster |
