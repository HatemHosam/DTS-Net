# DTS-Net: Depth-to-Space Networks for Fast and Accurate Semantic Object Segmentation
DTS-Net Depth-to-Space Net (DTS-Net), an effective technique for semantic segmentation using the efficient sub-pixel convolutional neural network. This technique is inspired by depth-to-space (DTS) image reconstruction, which was originally used for image and video superresolution tasks, combined with a mask enhancement filtration technique based on multi-label classification, namely, Nearest Label Filtration. In the proposed technique, we employ depth-wise separable convolution-based architectures. We propose both a deep network, that is, DTS-Net, and a lightweight network, DTS-Net-Lite, for real-time semantic segmentation; these networks employ Xception and MobileNetV2 architectures as the feature extractors, respectively. In addition, we explore the joint semantic segmentation and depth estimation task and demonstrate that the proposed technique can efficiently perform both tasks simultaneously, outperforming state-of-art (SOTA) methods. We train and evaluate the performance of the proposed method on the PASCAL VOC2012, NYUV2, and CITYSCAPES benchmarks. Hence, we obtain high mean intersection over union (mIOU) and mean pixel accuracy (Pix.acc.) values using simple and lightweight convolutional neural network architectures of the developed networks. Notably, the proposed method outperforms SOTA methods that depend on encoder–decoder architectures, although our implementation and computations are far simpler.<br />
The paper is published at Sensors, avaiable in: https://www.mdpi.com/1424-8220/22/1/337 <br />
requirements: <br />
Tensorflow '2.5.0' <br />
opencv '4.5.1' <br />
Numpy '1.19.5' <br />

NYU depth V2 groudtruth depth maps and semantic segmentation maps are saved in custom formats for easier use with the provided code, they are available at this link: https://drive.google.com/drive/u/2/folders/1tviXcOM7ToxNjL1CbztsxVqs70l6adqY <br />


if you use this paper, please cite this paper:
H. Ibrahem, A. Salem, and H.-S. Kang, “DTS-Net: Depth-to-Space Networks for Fast and Accurate Semantic Object Segmentation,” Sensors, vol. 22, no. 1, p. 337, Jan. 2022, doi: 10.3390/s22010337.
