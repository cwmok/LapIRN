# Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks

This is the official Pytorch implementation of "Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks" (MICCAI 2020), written by Tony C. W. Mok and Albert C. S. Chung.

\*\* Please also check out our new conditional deformable image registration framework ([c-LapIRN - MICCAI2021](https://arxiv.org/abs/2106.12673)) at https://github.com/cwmok/Conditional_LapIRN, which enables precise control of the smoothness of the deformation field and rapid hyperparameter tuning. \*\*

## Prerequisites
- `Python 3.5.2+`
- `Pytorch 1.3.0 - 1.7.0`
- `NumPy`
- `NiBabel`

This code has been tested with `Pytorch 1.3.0` and GTX1080TI GPU.

## Inference
If you prefer diffeomorphic solutions, please try:
```
python Test_LapIRN_diff.py
```

If you prefer solutions with maximized registration accuracy, please try:
```
python Test_LapIRN_disp.py
```

## Train your own model
Step 1: Replace `/PATH/TO/YOUR/DATA` with the path of your training data. You may also need to implement your own data generator (`Dataset_epoch` in `Functions.py`).

Step 2: Change the `imgshape` variable (in `Train_LapIRN_diff.py` or `Train_LapIRN_disp.py`) to match the resolution of your data.

(Optional) Step 3: You may adjust the size of the model by manipulating the argument `--start_channel`.

Step 3: `python Train_LapIRN_diff.py` to train the LapIRN formulated with the stationary velocity field, or `python Train_LapIRN_disp.py` to train the LapIRN formulated with the displacement field.

## Publication
If you find this repository useful, please cite:
- **Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks**  
[Tony C. W. Mok](https://cwmok.github.io/ "Tony C. W. Mok"), Albert C. S. Chung  
MICCAI 2020. [eprint arXiv:2006.16148](https://arxiv.org/abs/2006.16148 "eprint arXiv:2006.16148")

## Notes on this repository
We changed the regularization term in the loss function. The original regularization term is computed on the normalized velocity/displacement field, which may cause bias to the short axis.

## Acknowledgment
Some codes in this repository are modified from [IC-Net](https://github.com/zhangjun001/ICNet) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).

###### Keywords
Keywords: Diffeomorphic Image Registration, Large Deformation, Laplacian Pyramid Networks, Convolutional neural networks
