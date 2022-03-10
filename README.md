# Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks

This is the official Pytorch implementation of "Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks" (MICCAI 2020), written by Tony C. W. Mok and Albert C. S. Chung.

\*\* Please also check out our new conditional deformable image registration framework ([c-LapIRN - MICCAI2021](https://arxiv.org/abs/2106.12673)) at https://github.com/cwmok/Conditional_LapIRN, which enables precise control on the smoothness of the deformation field and rapid hyperparameter tuning. \*\*

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

## (Example) Training on the preprocessed OASIS dataset with downsampled images
If you want to train on the preprocessed OASIS dataset in https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md. We have an example showing how to train on this dataset.
1. Download the preprocessed OASIS dataset, unzip it and put it in "Data/OASIS".
2. To train a new LapIRN model, `python Train_LapIRN_diff_resize.py` will create a LapIRN model trained on all cases (with images resized to (144, 160, 192) resolution) in the dataset.
3. To test the model, `python Test_LapIRN_diff_resize.py --modelpath {{pretrained_model_path}} --fixed ../Data/image_A_fullsize.nii.gz --moving ../Data/image_B_fullsize.nii.gz` will load the assigned model and register the image "image_A_fullsize.nii.gz" and "image_B_fullsize.nii.gz".

Note that the LapIRN model in `Train_LapIRN_diff_resize.py` is trained with downsampled images with size indicated in the variable `imgshape`. Feel free to adjust the image size by adjusting the variable `imgshape` in `Train_LapIRN_diff_resize.py`.

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
