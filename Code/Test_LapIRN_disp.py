import os, shutil, time
from argparse import ArgumentParser

import numpy as np
import SimpleITK as sitk
import torch

from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LapIRN_disp_fea7.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B.nii',
                    help="moving image")
opt = parser.parse_args()
print(opt)

savepath     = opt.savepath
fixed_path   = opt.fixed
moving_path  = opt.moving

imgTmp = sitk.ReadImage(fixed_path)
imgshape = imgTmp.GetSize()
print("imgshape: ", imgshape)
imgshape_4 = (int(imgshape[0] / 4), int(imgshape[1] / 4), int(imgshape[2] / 4))
imgshape_2 = (int(imgshape[0] / 2), int(imgshape[1] / 2), int(imgshape[2] / 2))
range_flow = 0.4

in_channel = 2
n_classes  = 3
isTrainLvl1    = True
isTrainLvl2    = True
isTrainLvl3    = False # why???

doNormalisation = False

if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel


def test():
    # imgshape_4 = (160 / 4, 192 / 4, 144 / 4)
    # imgshape_2 = (160 / 2, 192 / 2, 144 / 2)

    # why is train true ?
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel, is_train=isTrainLvl2, imgshape=imgshape_2, range_flow=range_flow, model_lvl1=model_lvl1).cuda()
    model      = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(in_channel, n_classes, start_channel, is_train=isTrainLvl3, imgshape=imgshape,   range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device   = torch.device("cuda" if use_cuda else "cpu")
    #support nii, nii.gz, and nrrd
    ext = ".nii.gz" if ".nii.gz" in fixed_path else (".nii" if ".nii" in fixed_path else ".nrrd" )

    fixedName  = fixed_path.split('/')[-1][:-len(ext)]
    movingName = moving_path.split('/')[-1][:-len(ext)]

    fixed_img = load_4D(fixed_path)
    moving_img = load_4D(moving_path)

    fixed_img  = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        #get displacement field
        F_X_Y = model(moving_img, fixed_img)
        # convert to array ??
        F_X_Y_cpu    = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        F_X_Y_cpuF2F = transform_unit_flow_to_flow(F_X_Y_cpu)
        print("displacement field  F_X_Y: ",F_X_Y.shape)
        print("displacement field  F_X_Y: ",F_X_Y_cpu.shape)
        print("displacement field  F_X_Y: ",F_X_Y_cpuF2F.shape)

        # get transformed image
        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
        print("mving image           X_Y: ", X_Y.shape)

        # if segmentation found, transform segmentation as welll
        seg_path = moving_path[:-len(ext)]+'_seg'+ext
        if os.path.isfile(seg_path):
           seg_img =  load_4D(seg_path)
           seg_img = torch.from_numpy(seg_img).float().to(device).unsqueeze(dim=0)
           transformedSeg      = transform(seg_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
           outputMovingSegPath = savepath + '/' + movingName + '_seg_warpped_moving'+ext
           save_img (transformedSeg              , outputMovingSegPath)

        # save result
        outputFixedPath     = savepath +'/'+ fixedName + ext
        outputDispFieldPath = savepath +'/'+ movingName  + '_warpped_flow'  + ext
        outputMovingPath    = savepath +'/'+ movingName  + '_warpped_moving'+ ext
        print("outputFixedPath     : " ,outputFixedPath )
        print("outputDispFieldPath : " ,outputDispFieldPath )
        print("outputMovingPath    : " ,outputMovingPath )

        shutil.copyfile(fixed_path , outputFixedPath)
        save_flow(F_X_Y_cpuF2F     , outputDispFieldPath)
        save_img (X_Y              , outputMovingPath)

        #evaluation:
        # compute dice

    print("Finished")


if __name__ == '__main__':
    #imgshape = (160, 192, 144)
    test()
