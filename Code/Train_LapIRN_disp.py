import glob
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data

from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=30001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=30001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 0 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/PATH/TO/YOUR/DATA',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=2000,
                    help="Number step for freezing the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "LDR_OASIS_NCC_unit_disp_add_reg_1_"

def train_lvl1():
    print("Training lvl1...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                        range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1+1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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

    while step <= iteration_lvl1:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
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

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                          range_flow=range_flow).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl1_?????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
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

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl2_?????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    # transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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

    while step <= iteration_lvl3:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Validation

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


imgshape = (160, 192, 144)
imgshape_4 = (160/4, 192/4, 144/4)
imgshape_2 = (160/2, 192/2, 144/2)

range_flow = 0.4
train_lvl1()
train_lvl2()
train_lvl3()
