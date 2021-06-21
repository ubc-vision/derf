# DeRF: Decomposed Radiance Fields
## Daniel Rebain, Wei Jiang, Soroosh Yazdani, Ke Li, Kwang Moo Yi, Andrea Tagliasacchi

### Links
 - [Paper](https://arxiv.org/pdf/2011.12490.pdf)
 - [Project Page](https://ubc-vision.github.io/derf/)

### Abstract
With the advent of Neural Radiance Fields (NeRF), neural networks can now render novel views of a 3D scene with quality that fools the human eye. Yet, generating these images is very computationally intensive, limiting their applicability in practical scenarios. In this paper, we propose a technique based on spatial decomposition capable of mitigating this issue. Our key observation is that there are diminishing returns in employing larger (deeper and/or wider) networks. Hence, we propose to spatially decompose a scene and dedicate smaller networks for each decomposed part. When working together, these networks can render the whole scene. This allows us near-constant inference time regardless of the number of decomposed parts. Moreover, we show that a Voronoi spatial decomposition is preferable for this purpose, as it is provably compatible with the Painter’s Algorithm for efficient and GPU-friendly rendering. Our experiments show that for real-world scenes, our method provides up to 3x more efficient inference than NeRF (with the same rendering quality), or an improvement of up to 1.0~dB in PSNR (for the same inference cost).

### This Repository
This is the open-source code release for our paper "DeRF: Decomposed Radiance Fields".
Please note that this repository is provided as a one-time release, and is not being updated or maintained.
If you believe that there is a major issue that the authors need to be aware of, please contact us via email.

### Launch Commands
    python train.py experiment_name llff_fern
    python eval.py experiment_name llff_fern eval_results_dir
