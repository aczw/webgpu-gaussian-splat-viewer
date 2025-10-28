**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 5**

- Charles Wang
  - [LinkedIn](https://linkedin.com/in/zwcharl)
  - [Personal website](https://charleszw.com)
- Tested on:
  - Windows 11 Pro (26200.6899)
  - Ryzen 5 7600X @ 4.7Ghz
  - 32 GB RAM
  - RTX 5060 Ti 16 GB (Studio Driver 581.29)

# WebGPU Gaussian Splat Viewer

| [![](images/preview.png)](https://aczw.github.io/webgpu-gaussian-splat-viewer) |
| :----------------------------------------------------------------------------: |
|                      Bicycle scene, rendered at 1920×1080                      |

## Demo

A live version of this project is available at [aczw.github.io/webgpu-gaussian-splat-viewer](https://aczw.github.io/webgpu-gaussian-splat-viewer). Note that you'll have to provide your own scene and camera file. Below is a screen recording of the viewer in action.

https://github.com/user-attachments/assets/a519f247-0154-4d76-97e9-85a9719e21f0

## Overview

This project aims to implement the [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting) paper in WebGPU. It specifically focuses on the rasterization pipeline of the splats, meaning I did not concern myself with the training portion or creation of these gaussians.

I heavily referenced the original paper's rasterization engine, available at [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), for this project. It is written in C++ and CUDA; notably, it does not use a graphics API, meaning my implementation could not be a 1:1 translation.

The end result is an online viewer to load (theoretically) any gaussian splatting scene given the PLY file and at least one camera. For instance, let's grab the banana PLY file linked in the [OpenSplat](https://github.com/pierotofy/opensplat) README and try viewing it:

|    ![](images/banana.png)     |
| :---------------------------: |
| Banana, rendered at 1920×1080 |

## Implementation

To draw a single frame, we perform three main steps.

1. Preprocess compute shader pass
2. Radix sort compute shader pass
3. Splat data render pass

Below I go into more detail about each step.

### Preprocess compute pass

The preprocessing performs three major tasks. First, it ingests the raw gaussian data and computes necessary information used during rasterization. Second, we store some additional data in buffers to be used in the sorting pass. Third, we perform view frustum culling to reduce the number of computations and splats we'll need to draw in the end.

### Sorting the splats

### Rasterization

## Performance analysis

### Questions

> Compare your results from point-cloud and gaussian renderer, what are the differences?

> For gaussian renderer, how does changing the workgroup-size affect performance? Why do you think this is?

> Does view-frustum culling give performance improvement? Why do you think this is?

> Does number of guassians affect performance? Why do you think this is?

## Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special thanks to Shrek Shao (from the Google WebGPU team) and the original [differential gaussian rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization) implementation
