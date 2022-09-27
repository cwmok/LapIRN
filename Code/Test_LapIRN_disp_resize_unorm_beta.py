import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from Functions import generate_grid, save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1, Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2, \
    Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3, SpatialTransform

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LDR_OASIS_NCC_unit_add_reg_resize_35_stagelvl3_5000.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=6,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/OASIS/OASIS_OAS1_0001_MR1/aligned_norm.nii.gz',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/OASIS/OASIS_OAS1_0002_MR1/aligned_norm.nii.gz',
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel



def test():
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    # grid = generate_grid_unit(imgshape)
    # grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_full = generate_grid(ori_imgshape)
    grid_full = torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    fixed_img = load_4D(fixed_path)
    moving_img = load_4D(moving_path)

    # get header
    temp = nib.load(fixed_path)
    header, affine = temp.header, temp.affine

    # normalize image to [0, 1]
    norm = True
    if norm:
        fixed_img = imgnorm(fixed_img)
        moving_img = imgnorm(moving_img)

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        moving_img_down = F.interpolate(moving_img, size=imgshape, mode='trilinear')
        fixed_img_down = F.interpolate(fixed_img, size=imgshape, mode='trilinear')

        F_X_Y = model(moving_img_down, fixed_img_down)
        F_X_Y = F.interpolate(F_X_Y, size=ori_imgshape, mode='trilinear', align_corners=True)

        scale = [x[1] / x[0] for x in zip(imgshape, ori_imgshape)]
        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * scale[2]
        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * scale[1]
        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * scale[0]

        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid_full).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.flip(1).data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        save_flow(F_X_Y_cpu, savepath+'/warpped_flow.nii.gz', header=header, affine=affine)
        save_img(X_Y, savepath+'/warpped_moving.nii.gz', header=header, affine=affine)

    print("Finished")


if __name__ == '__main__':
    ori_imgshape = (160, 192, 224)
    imgshape = (144, 160, 192)
    imgshape_4 = (144 // 4, 160 // 4, 192 // 4)
    imgshape_2 = (144 // 2, 160 // 2, 192 // 2)
    range_flow = 60
    test()
