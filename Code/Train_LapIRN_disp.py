import glob, os, sys, math
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data

from argparse import ArgumentParser
from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid, saveLog, iaLog2Fig, check_metric, load_4D,diceMetric
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
TODOs:
    - create multiple figures, function that create figures from the log file.  
    - colab: download the data and start from last epoch ---------------------->done! not working
    - add more comments and explanations  
      - study the models architecture                
    - add logging ------------------------> done!
    - add testing ------------------------> done!
    - dice and mse evaluation 
    - add multiple testing functions  
    - add augmentation
    - improve data generator
      - study the load manager code
    - add dice loss
'''

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--sIteration_lvl1", type=int,
                    dest="sIteration_lvl1", default=0,
                    help="start of lvl1 iterations")
parser.add_argument("--sIteration_lvl2", type=int,
                    dest="sIteration_lvl2", default=0,
                    help="start of lvl2 iterations")
parser.add_argument("--sIteration_lvl3", type=int,
                    dest="sIteration_lvl3", default=0,
                    help="start of lvl3 iterations")
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
start_channel = 8 #opt.start_channel
antifold      = 0.0 # opt.antifold
smooth       =  2.0 # opt.smooth

n_checkpoint  = opt.checkpoint
print("n_checkpoint : ",n_checkpoint)
datapath     = opt.datapath
print("datapath : ",datapath)

# the dataset folder is modified to be img<id>.nii.gz and img<id>_seg.nii.gz in the same folder
# datasets:
# 1. L2R_Task3_AbdominalCT:
# 2. OASIS

#names = sorted(glob.glob(datapath + '/*.nii'))[0:255]
fnms      = sorted(os.listdir(datapath))
names    = [os.path.join(datapath,x) for x in fnms if not "seg" in x]
namesSeg = [os.path.join(datapath,x) for x in fnms if     "seg" in x]
trainingLst = names[:-5]
testingLst  = names[-5:]
print((datapath))
print("len(trainingLst) : ",len(trainingLst))
print("len(testingLst)  : ",len(testingLst))

#imgshape   = (160, 192, 144) # OASIS
#imgshape   = (192, 160, 256)  # OASIS
imgTmp = sitk.ReadImage(trainingLst[0])
imgshape   = imgTmp.GetSize()
print("imgshape: ",imgshape)
imgshape_4 = (int(imgshape[0]/4), int(imgshape[1]/4), int(imgshape[2]/4))
imgshape_2 = (int(imgshape[0]/2), int(imgshape[1]/2), int(imgshape[2]/2))
range_flow = 0.4
in_channel = 2
n_classes  = 3
isTrainLvl1    = True # #TODO:  ???
isTrainLvl2    = True
isTrainLvl3    = True

doNormalisation = False
#model_folder_path  = "../Model/Stage"
model_dir = '../Model/Stage'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

result_folder_path = "../Results"
logLvl1Path        = model_dir + "/logLvl1.txt"
logLvl2Path        = model_dir + "/logLvl2.txt"
logLvl3Path        = model_dir + "/logLvl3.txt"
logLvl1ChrtPath    = model_dir + "/logLvl1.png"
logLvl2ChrtPath    = model_dir + "/logLvl2.png"
logLvl3ChrtPath    = model_dir + "/logLvl3.png"

freeze_step = opt.freeze_step #TODO:  ???
numWorkers = 0 #2  number of threads for the data generators???
print(opt)
sIteration_lvl1 = opt.sIteration_lvl1
sIteration_lvl2 = opt.sIteration_lvl2
sIteration_lvl3 = opt.sIteration_lvl3

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3
load_model_lvl1 = True if sIteration_lvl1>0 else False
load_model_lvl2 = True if sIteration_lvl2>0 else False
load_model_lvl3 = True if sIteration_lvl3>0 else False

model_name = "LDR_OASIS_NCC_unit_disp_add_reg_1_"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_lvl1_path = model_dir + '/' + model_name + "stagelvl1_0.pth"
loss_lvl1_path  = model_dir + '/loss' + model_name + "stagelvl1_0.npy"
model_lvl2_path = model_dir + '/' + model_name + "stagelvl2_0.pth"
loss_lvl2_path  = model_dir + '/loss' + model_name + "stagelvl2_0.npy"
model_lvl3_path = model_dir + '/' + model_name + "stagelvl3_0.pth"
loss_lvl3_path  = model_dir + '/loss' + model_name + "stagelvl3_0.npy"

# log1= '/home/ibr/Downloads/Stage-20210421T142530Z-001/logLvl1.txt'
# log2= log1[:-5]+'2.txt'
# log3= log1[:-5]+'3.txt'
# iaLog2Fig(log1)
# iaLog2Fig(log2)
# iaLog2Fig(log3)
# print(ok)

def train_lvl1():
    print("Training lvl1...")
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    loss_Jdet       = neg_Jdet_loss
    loss_smooth     = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False # TODO: ???
        param.volatile      = True  # TODO: ???

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    lossall = np.zeros((4, iteration_lvl1+1))

    #TODO: improve the data generator:
    #  - use fixed lists for training and testing
    #  - use augmentation

    training_generator = Data.DataLoader(Dataset_epoch(trainingLst, norm=doNormalisation), batch_size=1,shuffle=True, num_workers=numWorkers)

    step = 0
    if sIteration_lvl1>0:
        # model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        # loss_path   = "../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy"
        model_lvl1_path  =  model_dir + '/'     + model_name + "stagelvl1_" + str(sIteration_lvl1) + '.pth'
        loss_lvl1_path   =  model_dir + '/loss' + model_name + "stagelvl1_" + str(sIteration_lvl1) + '.npy'
        print("Loading weight and loss : ", model_lvl1_path)
        step = sIteration_lvl1+1
        model.load_state_dict(torch.load(model_lvl1_path))
        temp_lossall = np.load(loss_lvl1_path)
        lossall[:, 0:sIteration_lvl1] = temp_lossall[:, 0:sIteration_lvl1]
    else:
        #create log file only when
        logLvl1File = open(logLvl1Path, "w") ;  logLvl1File.close

    stepsLst = [] ; lossLst= [] ; simNCCLst = []; JdetLst=[] ; smoLst= []

    # for each iteration
    #TODO: modify the iteration to be related to the number of images
    while step <= iteration_lvl1:
        #for each pair in the data generator
        for pair in training_generator:
            X = pair[0][0];
            Y = pair[0][1]
            movingPath = pair[1][0]
            fixedPath  = pair[1][1]

            X = X.to(device).float()
            Y = Y.to(device).float()
            assert not np.any(np.isnan(X.cpu().numpy()))
            assert not np.any(np.isnan(Y.cpu().numpy()))

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            # F_X_Y: displacement_field,
            # X_Y: wrapped_moving_image,
            # Y_4x: downsampled_fixed_image,
            # F_xy: velocity_field
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)
            F_X_Y_norm    = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)
            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * z
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * y
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * x
            loss_regulation = loss_smooth(F_X_Y)

            assert not np.any(np.isnan(loss_multiNCC.item() ))
            assert not np.any(np.isnan(loss_Jacobian.item()))
            assert not np.any(np.isnan(loss_regulation.item()))

            #TODO: ??? try o to make more influence to the similarity weight
            #TODO: ??? antifold?
            loss = loss_multiNCC    +  antifold * loss_Jacobian    +      smooth * loss_regulation

            assert not np.any(np.isnan(loss.item()))

            # TODO: ??? why clearing optimiser evey new example?
            optimizer.zero_grad()           # clear gradients for this training step

            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            logLine = "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item())
            #sys.stdout.write(logLine)
            #sys.stdout.flush()
            print(logLine)
            # save log:
            #saveLog(logLvl1Path,logLine)
            logLvl1File = open(logLvl1Path,"a");  logLvl1File.write(logLine);  logLvl1File.close()
            iaLog2Fig(logLvl1Path)
            # stepsLst.append(step); lossLst.append(loss.item()); simNCCLst.append(loss_multiNCC.item()) ; JdetLst.append(loss_Jacobian.item()); smoLst.append(loss_regulation.item())
            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                model_lvl1_path = model_dir + '/'     + model_name + "stagelvl1_" + str(step) + '.pth'
                loss_lvl1_path  = model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy'
                torch.save(model.state_dict(), model_lvl1_path)
                np.save(loss_lvl1_path, lossall)

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass ....")
    model_lvl1_path = model_dir + '/'     + model_name + "stagelvl1_" + str(iteration_lvl1) + '.pth'
    loss_lvl1_path  = model_dir + '/loss' + model_name + "stagelvl1_" + str(iteration_lvl1) + '.npy'
    torch.save(model.state_dict(), model_lvl1_path)
    np.save(loss_lvl1_path, lossall)
    return model_lvl1_path


def train_lvl2(model_lvl1_path):
    print("Training lvl2...")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).to(device)
    print("Loading weight for model_lvl1...", model_lvl1_path)
    model_lvl1.load_state_dict(torch.load(model_lvl1_path))

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel, is_train=isTrainLvl2, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet  = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((4, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(trainingLst, norm=doNormalisation), batch_size=1,
                                         shuffle=True, num_workers=numWorkers)
    step = 0
    if sIteration_lvl2>0:
        model_lvl2_path  =  model_dir + '/'     + model_name + "stagelvl2_" + str(sIteration_lvl2) + '.pth'
        loss_lvl2_path   =  model_dir + '/loss' + model_name + "stagelvl2_" + str(sIteration_lvl2) + '.npy'
        print("Loading weight and loss : ", model_lvl2_path)
        step = sIteration_lvl2+1
        model.load_state_dict(torch.load(model_lvl2_path))
        temp_lossall = np.load(loss_lvl2_path)
        lossall[:, 0:sIteration_lvl2] = temp_lossall[:, 0:sIteration_lvl2]
    else:
        #create log file
        logLvl2File = open(logLvl2Path, "w") ;  logLvl2File.close

    stepsLst = [];    lossLst = [];    simNCCLst = [];    JdetLst = [];    smoLst = []

    while step <= iteration_lvl2:
        for pair in training_generator:
            X = pair[0][0];
            Y = pair[0][1]
            movingPath = pair[1][0]
            fixedPath  = pair[1][1]

            X = X.to(device).float()
            Y = Y.to(device).float()

            assert not np.any(np.isnan(X.cpu().numpy()  ))
            assert not np.any(np.isnan(Y.cpu().numpy()  ))


            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * z
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * y
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * x
            loss_regulation = loss_smooth(F_X_Y)

            assert not np.any(np.isnan(loss_multiNCC.item() ))
            assert not np.any(np.isnan(loss_Jacobian.item()))
            assert not np.any(np.isnan(loss_regulation.item()))

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            assert not np.any(np.isnan(loss.item()))

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            logLine = "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item())
            # sys.stdout.write(logLine)
            # sys.stdout.flush()
            print(logLine)

            # save log:
            logLvl2File = open(logLvl2Path,"a");  logLvl2File.write(logLine);  logLvl2File.close()
            iaLog2Fig(logLvl2Path)

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                model_lvl2_path = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                loss_lvl2_path = model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy'
                torch.save(model.state_dict(), model_lvl2_path)
                np.save(loss_lvl2_path, lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
        model_lvl2_path = model_dir + '/' + model_name + "stagelvl2_" + str(iteration_lvl2) + '.pth'
        loss_lvl2_path = model_dir + '/loss' + model_name + "stagelvl2_" + str(iteration_lvl2) + '.npy'
        torch.save(model.state_dict(), model_lvl2_path)
        np.save(loss_lvl2_path, lossall)
    return model_lvl2_path


def train_lvl3(model_lvl1_path , model_lvl2_path):
    print("Training lvl3...")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel, is_train=isTrainLvl2, imgshape=imgshape_2, range_flow=range_flow,model_lvl1=model_lvl1).to(device)
    print("Loading weight for model_lvl2...", model_lvl2_path)
    model_lvl2.load_state_dict(torch.load(model_lvl2_path))

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(in_channel, n_classes, start_channel, is_train=isTrainLvl3, imgshape=imgshape, range_flow=range_flow, model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform         = SpatialTransform_unit().to(device)
    #TODO: not used ???
    #transform_nearest = SpatialTransformNearest_unit().to(device)
    transform_nearest = SpatialTransform_unit('nearest').to(device)
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    #grid_unit = generate_grid_unit(imgshape)
    grid_unit = generate_grid(imgshape,1)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((4, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(trainingLst, norm=doNormalisation), batch_size=1,  shuffle=True, num_workers=numWorkers)
    step = 0
    if sIteration_lvl3>0:
        model_lvl3_path  =  model_dir + '/'     + model_name + "stagelvl3_" + str(sIteration_lvl3) + '.pth'
        loss_lvl3_path   =  model_dir + '/loss' + model_name + "stagelvl3_" + str(sIteration_lvl3) + '.npy'
        print("Loading weight and loss : ", model_lvl3_path)
        step = sIteration_lvl3+1
        model.load_state_dict(torch.load(model_lvl3_path))
        temp_lossall = np.load(loss_lvl3_path)
        lossall[:, 0:sIteration_lvl3] = temp_lossall[:, 0:sIteration_lvl3]
    else:
        logLvl3File = open(logLvl3Path, "w");    logLvl3File.close()

    stepsLst = [];    lossLst = [];    simNCCLst = [];    JdetLst = [];    smoLst = []
    total_avg_dice = 0.0;
    while step <= iteration_lvl3:
        for pair in training_generator:
            X = pair[0][0];
            Y = pair[0][1]
            moving_img_Path = pair[1][0][0]
            fixed_img_path  = pair[1][1][0]

            X = X.to(device).float()
            Y = Y.to(device).float()
            assert not np.any(np.isnan(X.cpu().numpy()  ))
            assert not np.any(np.isnan(Y.cpu().numpy()  ))

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * z
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * y
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * x
            loss_regulation = loss_smooth(F_X_Y)

            assert not np.any(np.isnan(loss_multiNCC.item() ))
            assert not np.any(np.isnan(loss_Jacobian.item()))
            assert not np.any(np.isnan(loss_regulation.item()))

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            assert not np.any(np.isnan(loss.item()))

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            lossall[:, step] = np.array([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            logLine = "\r" + 'step '+ str(step)+" -> training loss " + str(loss.item()) + " - sim_NCC " +str( loss_multiNCC.item())+" - Jdet "+str( loss_Jacobian.item())+" -smo "+str( loss_regulation.item())+" -dice "+ str(total_avg_dice)
            # sys.stdout.write(logLine)
            # sys.stdout.flush()
            print(logLine)
            # save log:
            # saveLog(logLvl1Path,logLine)
            logLvl3File = open(logLvl3Path, "a"); logLvl3File.write(logLine);         logLvl3File.close()
            iaLog2Fig(logLvl3Path)
            # stepsLst.append(step); lossLst.append(loss.item()); simNCCLst.append(loss_multiNCC.item()) ; JdetLst.append(loss_Jacobian.item()); smoLst.append(loss_regulation.item())
            # with lr 1e-3 + with bias
            n_checkpoint = 10
            if (step % n_checkpoint == 0):
            #if True:
                # save the current model
                model_lvl3_path = model_dir + '/'     + model_name + "stagelvl3_" + str(step) + '.pth'
                loss_lvl3_path  = model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy'
                torch.save(model.state_dict(), model_lvl3_path)
                np.save(loss_lvl3_path, lossall)

                testCounter = 0.0
                testing_generator = Data.DataLoader(Dataset_epoch(testingLst, norm=doNormalisation), batch_size=1,  shuffle=True, num_workers=numWorkers)
                for test_pair in testing_generator:
                    tX = test_pair[0][0];
                    tY = test_pair[0][1]
                    moving_img_Path = test_pair[1][0][0]
                    fixed_img_path  = test_pair[1][1][0]
                    # print(fixed_img_path)
                    # print(moving_img_Path)
                    fixed_seg_path  = fixed_img_path[:-7] + "_seg.nii.gz"
                    moving_seg_Path = moving_img_Path[:-7] + "_seg.nii.gz"

                    fixed_seg  = sitk.GetArrayFromImage(sitk.ReadImage(fixed_seg_path))
                    fixed_seg  = np.swapaxes(fixed_seg,0,2)
                    #fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())

                    moving_seg = sitk.GetArrayFromImage(sitk.ReadImage(moving_seg_Path))

                    #convert to binary classes
                    fixed_seg[fixed_seg>0.0]     = 1.0
                    moving_seg[moving_seg > 0.0] = 1.0
                    # fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())

                    # print("len  len(np.unique(fixed_seg)) : ", len(np.unique(fixed_seg)))
                    # print("len  len(np.unique(moving_seg)): ", len(np.unique(moving_seg)))

                    #moving_seg = load_4D(moving_seg_Path)
                    moving_seg = moving_seg[np.newaxis,...]
                    moving_seg_tensor = torch.from_numpy(moving_seg).float().to(device).unsqueeze(dim=0)

                    tX = tX.to(device).float()
                    tY = tY.to(device).float()
                    # assert not np.any(np.isnan(X.cpu().numpy()))
                    # assert not np.any(np.isnan(Y.cpu().numpy()))

                    # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
                    with torch.no_grad():
                       disp_fld, tX_Y, tY_4x, tF_xy, tF_xy_lvl1, tF_xy_lvl2, _ = model(tX, tY)
                    #disp_fld, tX_Y, tY_4x, tF_xy, tF_xy_lvl1, tF_xy_lvl2, _ = model.eval(tX, tY)
                    transformed_seg_image = transform(moving_seg_tensor, disp_fld.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
                    transformed_seg_image[transformed_seg_image > 0.0] = 1.0

                    # print("fixed_seg                :",fixed_seg.shape)
                    # print("transformed_moving_image :",transformed_seg_image.shape)
                    # print("len  len(np.unique(transformed_seg_image)): ", len(np.unique(transformed_seg_image)))
                    # computet he dice
                    gt = fixed_seg.ravel();
                    res = transformed_seg_image.ravel()
                    total_avg_dice = total_avg_dice + np.sum(res[gt == 1]) * 2.0 / (np.sum(res) + np.sum(gt))
                    testCounter+=1
                total_avg_dice = total_avg_dice / testCounter

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    model_lvl3_path = model_dir + '/'    + model_name + "stagelvl3_" + str(iteration_lvl3) + '.pth'
    loss_lvl3_path  = model_dir + '/loss' + model_name + "stagelvl3_" + str(iteration_lvl3) + '.npy'
    torch.save(model.state_dict(), model_lvl3_path)
    np.save(loss_lvl3_path, loss_lvl3_path)


# ................................... Start training  .....................................

model_lvl1_path = train_lvl1()
model_lvl2_path = train_lvl2(model_lvl1_path)
model_lvl3_path = train_lvl3(model_lvl1_path , model_lvl2_path)
