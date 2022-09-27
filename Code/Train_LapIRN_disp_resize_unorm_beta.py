import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, Dataset_epoch_validation
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1, Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2, \
    Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3, SpatialTransform, SpatialTransformNearest, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=20001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=20001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
# parser.add_argument("--antifold", type=float,
#                     dest="antifold", default=0.,
#                     help="Anti-fold loss: suggested range 0 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=1000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=6,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../Data/OASIS',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=2000,
                    help="Number step for freezing the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
# antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "LDR_OASIS_NCC_disp_add_reg_resize_1_"


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def train_lvl1():
    print("Training lvl1...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl1+1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=4)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            # loss_Jacobian = loss_Jdet(F_X_Y.permute(0,2,3,4,1), grid_4)

            # reg2 - use velocity
            # _, _, x, y, z = F_xy.shape
            # F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z-1)
            # F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y-1)
            # F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            # loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                          range_flow=range_flow).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl1_?????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')

            # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            # loss_Jacobian = loss_Jdet(F_X_Y.permute(0,2,3,4,1), grid_2)

            # reg2 - use velocity
            # _, _, x, y, z = F_xy.shape
            # F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z-1)
            # F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y-1)
            # F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            # loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl2_?????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().to(device)
    transform_nearest = SpatialTransformNearest().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    grid_full = generate_grid(ori_imgshape)
    grid_full = torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape)).to(device).float()

    # grid_unit = generate_grid_unit(ori_imgshape)
    # grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
        print("Loading weight: ", model_path)
        step = 10000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy")
        lossall[:, 0:10000] = temp_lossall[:, 0:10000]

    while step <= iteration_lvl3:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')

            # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            # loss_Jacobian = loss_Jdet(F_X_Y.permute(0,2,3,4,1), grid)

            # reg2 - use velocity
            # _, _, x, y, z = F_xy.shape
            # F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z-1)
            # F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y-1)
            # F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            # loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Validation
                # Put your validation code here
                # ---------------------------------------
                # OASIS (Validation)
                imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[-4:]
                labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))[-4:]

                valid_generator = Data.DataLoader(
                    Dataset_epoch_validation(imgs, labels, norm=True),
                    batch_size=1,
                    shuffle=False, num_workers=2)

                dice_total = []
                use_cuda = True
                device = torch.device("cuda" if use_cuda else "cpu")
                print("\nValiding...")
                for batch_idx, data in enumerate(valid_generator):
                    X, Y, X_label, Y_label = data[0].to(device), data[1].to(device), data[2].to(
                        device), data[3].to(device)

                    X = F.interpolate(X, size=imgshape, mode='trilinear')
                    Y = F.interpolate(Y, size=imgshape, mode='trilinear')

                    with torch.no_grad():
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)
                        F_X_Y = F.interpolate(F_X_Y, size=ori_imgshape, mode='trilinear', align_corners=True)

                        scale = [x[1]/x[0] for x in zip(imgshape, ori_imgshape)]
                        F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * scale[2]
                        F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * scale[1]
                        F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * scale[0]

                        X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_full).cpu().numpy()[0,
                                    0, :, :, :]
                        Y_label = Y_label.cpu().numpy()[0, 0, :, :, :]

                        dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                        dice_total.append(dice_score)

                print("Dice mean: ", np.mean(dice_total))
                with open(log_dir, "a") as log:
                    log.write(str(step) + ":" + str(np.mean(dice_total)) + "\n")

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == "__main__":
    ori_imgshape = (160, 192, 224)
    imgshape = (144, 160, 192)
    imgshape_4 = (144 // 4, 160 // 4, 192 // 4)
    imgshape_2 = (144 // 2, 160 // 2, 192 // 2)

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

    range_flow = 60
    start_t = datetime.now()
    train_lvl1()
    train_lvl2()
    train_lvl3()
    # time
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
