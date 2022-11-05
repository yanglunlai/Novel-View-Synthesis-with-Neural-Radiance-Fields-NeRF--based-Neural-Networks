# Novel View Synthesis with Neural Radiance Fields Neural Networks
University of Michigan EECS598 Deep Learning for Computer Vision Final Project

Instructor: Justin Johnson

2022 Winter

## Introduction
The Novel View Synthesis technique is used to synthesize a new scene of a target image with an arbitrary target camera pose based on given source images and their camera poses. In 2020, a novel method, Neural Radiance Fields (NeRF), achieved good results synthesizing unknown views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views [1]. From then on, NeRF-based algorithms have become a research hot-pot in the deep learning and computer vision field.

One of the significant challenges of the NeRF-based algorithms is the high computational expense for rendering images. Some methods, such as KiloNeRF [2], SNERG [3], and FastNeRF [4], solved this issue by utilizing new model architectures or novel representations to accelerate NeRF. In this project, we aim to implement NeRF-based models, generate results based on the NeRF dataset, and compare them with the original NeRF paper. We implement the models proposed in NeRF [1] and FastNeRF [4] and accelerate the training process by using smaller MLP, no hierarchical sampling, and lower resolution images.

## References

[1]
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Computer Vision – ECCV 2020, Cham: Springer International Publishing, 2020, pp. 405–421.

[2]
C. Reiser, S. Peng, Y. Liao, and A. Geiger, “KiloNeRF: Speeding up neural radiance fields with thousands of tiny MLPs,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[3]
P. Hedman, P. P. Srinivasan, B. Mildenhall, J. T. Barron, and P. Debevec, “Baking neural radiance fields for real-time view synthesis,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[4]
S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, and J. Valentin, “FastNeRF: High-fidelity neural rendering at 200FPS,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
