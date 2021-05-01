import glob, os, sys, math
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.transform import resize
#import skimage as ski
import torch
import torch.utils.data as Data

from argparse import ArgumentParser
from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid, saveLog, iaLog2Fig, check_metric, load_4D,diceMetric,img2SegTensor
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC,mseLoss,diceLoss,DiceLoss,mse_loss
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
parser.add_argument("--simLossType", type=int,
                    dest="simLossType", default=0,
                    help="type of loss: 0=NCC, 1=MSE, 2=Dice")

opt = parser.parse_args()

doValidation = 0

lr = opt.lr
start_channel = opt.start_channel
antifold      = opt.antifold
smooth       =  opt.smooth
datapath     = opt.datapath

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

numWorkers = 0 #2  number of threads for the data generators???


# log1= '/home/ibr/Downloads/Stage-20210421T142530Z-001/logLvl1.txt'
# log2= log1[:-5]+'2.txt'
# log3= log1[:-5]+'3.txt'
# iaLog2Fig(log1)
# iaLog2Fig(log2)
# iaLog2Fig(log3)
# print(ok)

def train(lvlID , opt=[], model_lvl1_path="" , model_lvl2_path=""):
    print("Training " + str(lvlID) +"===========================================================" )
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    freeze_step = opt.freeze_step  # TODO:  ???


    model_name = "LDR_OASIS_NCC_unit_disp_add_reg_1_"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl_path = model_dir + '/'    + model_name + "stagelvl"+str(lvlID)+"_0.pth"
    loss_lvl_path = model_dir + '/loss' + model_name + "stagelvl"+str(lvlID)+"_0.npy"

    n_checkpoint =  opt.checkpoint

    if lvlID==1:
        model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).to(device)
        grid = generate_grid(imgshape_4)

        start_iteration = opt.sIteration_lvl1
        num_iteration =  opt.iteration_lvl1
        logLvlPath = logLvl1Path
    elif lvlID==2:
        model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1, imgshape=imgshape_4, range_flow=range_flow).to(device)
        model_lvl1.load_state_dict(torch.load(model_lvl1_path))
        # Freeze model_lvl1 weight
        for param in model_lvl1.parameters():
            param.requires_grad = False

        model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel, is_train=isTrainLvl2, imgshape=imgshape_2,   range_flow=range_flow, model_lvl1=model_lvl1).to(device)

        grid = generate_grid(imgshape_2)
        start_iteration = opt.sIteration_lvl2
        num_iteration =  opt.iteration_lvl2
        logLvlPath = logLvl2Path
    elif lvlID==3:
        model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel,is_train=isTrainLvl1, imgshape=imgshape_4,  range_flow=range_flow).to(device)
        model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel,is_train=isTrainLvl2, imgshape=imgshape_2,range_flow=range_flow, model_lvl1=model_lvl1).to(device)
        model_lvl2.load_state_dict(torch.load(model_lvl2_path))
        # Freeze model_lvl1 weight
        for param in model_lvl2.parameters():
            param.requires_grad = False

        model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(in_channel, n_classes, start_channel, is_train=isTrainLvl3, imgshape=imgshape, range_flow=range_flow, model_lvl2=model_lvl2).to(device)

        grid = generate_grid(imgshape)
        start_iteration = opt.sIteration_lvl3
        num_iteration =  opt.iteration_lvl3
        logLvlPath = logLvl3Path


    load_model_lvl = True if start_iteration > 0 else False

    loss_Jdet       = neg_Jdet_loss
    loss_smooth     = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False # TODO: ???
        param.volatile      = True  # TODO: ???


    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    lossall = np.zeros((4, num_iteration+1))

    #TODO: improve the data generator:
    #  - use fixed lists for training and testing
    #  - use augmentation

    training_generator = Data.DataLoader(Dataset_epoch(trainingLst, norm=doNormalisation), batch_size=1,shuffle=True, num_workers=numWorkers)

    step = 0
    if start_iteration>0:
            # model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
            # loss_path   = "../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy"
            model_lvl_path  =  model_dir + '/'     + model_name + "stagelvl" + str(lvlID) + "_" + str(num_iteration) + '.pth'
            loss_lvl_path   =  model_dir + '/loss' + model_name + "stagelvl" + str(lvlID) +  "_" + str(num_iteration) + '.npy'
            print("Loading weight and loss : ", model_lvl_path)
            step = num_iteration+1
            model.load_state_dict(torch.load(model_lvl_path))
            temp_lossall = np.load(loss_lvl_path)
            lossall[:, 0:num_iteration] = temp_lossall[:, 0:num_iteration]
    else:
        #create log file only when
        logLvlFile = open(logLvlPath, "w") ;  logLvlFile.close

    stepsLst = [] ; lossLst= [] ; simNCCLst = []; JdetLst=[] ; smoLst= []

    # for each iteration
    #TODO: modify the iteration to be related to the number of images
    while step <= num_iteration:
        #for each pair in the data generator
        for pair in training_generator:
            X = pair[0][0];
            Y = pair[0][1]
            movingPath = pair[1][0][0]
            fixedPath  = pair[1][1][0]
            ext = ".nii.gz" if ".nii.gz" in fixedPath else (".nii" if ".nii" in fixedPath else ".nrrd")

            X = X.to(device).float()
            Y = Y.to(device).float()
            assert not np.any(np.isnan(X.cpu().numpy()))
            assert not np.any(np.isnan(Y.cpu().numpy()))

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            # F_X_Y: displacement_field,
            # X_Y: wrapped_moving_image,
            # Y_4x: downsampled_fixed_image,
            # F_xy: velocity_field
            if lvlID==1:
               F_X_Y, X_Y, Y_4x, F_xy, _                       = model(X, Y)
            elif lvlID==2:
               F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _            = model(X, Y)
            elif lvlID==3:
               F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # print("Y_4x shape : ",Y_4x.shape)
            # print("X_Y shape  : ",X_Y.shape)
            if opt.simLossType == 0:  # NCC
                if lvlID==1:
                   loss_similarity = NCC(win=3)
                elif lvlID==2:
                    loss_similarity = multi_resolution_NCC(win=5, scale=2)
                elif lvlID==3:
                    loss_similarity = multi_resolution_NCC(win=7, scale=3)
                loss_sim        = loss_similarity(X_Y, Y_4x)
            elif opt.simLossType == 1:  # mse loss
                #loss_sim = mseLoss(X_Y, Y_4x)
                loss_sim = mse_loss(X_Y, Y_4x)
                #print("loss_sim : ",loss_sim)
            elif opt.simLossType == 2:  # Dice loss
                # transform seg
                dv = math.pow(2, 3 - lvlID)
                fixedSeg  = img2SegTensor(fixedPath, ext,  dv)
                movingSeg = img2SegTensor(movingPath, ext, dv)

                movingSeg =movingSeg[np.newaxis,...]
                movingSeg = torch.from_numpy(movingSeg).float().to(device).unsqueeze(dim=0)
                transformedSeg = transform(movingSeg, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
                transformedSeg[transformedSeg>0]=1.0 ;
                loss_sim = diceLoss(fixedSeg, transformedSeg)
                loss_sim = DiceLoss.getDiceLoss(fixedSeg, transformedSeg)
            else:
                print("error: not supported loss ........")

            # 3 level deep supervision NCC
            F_X_Y_norm    = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)
            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * z
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * y
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * x
            loss_regulation = loss_smooth(F_X_Y)

            assert not np.any(np.isnan(loss_sim.item() ))
            assert not np.any(np.isnan(loss_Jacobian.item()))
            assert not np.any(np.isnan(loss_regulation.item()))

            loss = loss_sim    +  antifold * loss_Jacobian    +      smooth * loss_regulation

            assert not np.any(np.isnan(loss.item()))

            # TODO: ??? why clearing optimiser evey new example?
            optimizer.zero_grad()           # clear gradients for this training step

            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_sim.item(), loss_Jacobian.item(), loss_regulation.item()])

            logLine = "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(step, loss.item(), loss_sim.item(), loss_Jacobian.item(), loss_regulation.item())
            #sys.stdout.write(logLine)
            #sys.stdout.flush()
            print(logLine)
            # save log:
            #saveLog(logLvl1Path,logLine)
            logLvlFile = open(logLvlPath,"a");  logLvlFile.write(logLine);  logLvlFile.close()
            iaLog2Fig(logLvlPath)
            # stepsLst.append(step); lossLst.append(loss.item()); simNCCLst.append(loss_multiNCC.item()) ; JdetLst.append(loss_Jacobian.item()); smoLst.append(loss_regulation.item())
            # with lr 1e-3 + with bias
            if lvlID==3:
               n_checkpoint = 10

            if (step % n_checkpoint == 0):
                model_lvl_path = model_dir + '/'     + model_name + "stagelvl" + str(lvlID) + "_" + str( step) + '.pth'
                loss_lvl_path  = model_dir + '/loss' + model_name + "stagelvl" + str(lvlID) + "_" + str( step) + '.npy'
                torch.save(model.state_dict(), model_lvl_path)
                #np.save(loss_lvl_path, lossall)
                # if doValidation:
                #     # save the current model
                #     model_lvl3_path = model_dir + '/'     + model_name + "stagelvl3_" + str(step) + '.pth'
                #     loss_lvl3_path  = model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy'
                #     torch.save(model.state_dict(), model_lvl3_path)
                #     np.save(loss_lvl3_path, lossall)
                #
                #     testCounter = 0.0
                #     testing_generator = Data.DataLoader(Dataset_epoch(testingLst, norm=doNormalisation), batch_size=1,  shuffle=True, num_workers=numWorkers)
                #     for test_pair in testing_generator:
                #         tX = test_pair[0][0];
                #         tY = test_pair[0][1]
                #         moving_img_Path = test_pair[1][0][0]
                #         fixed_img_path  = test_pair[1][1][0]
                #         # print(fixed_img_path)
                #         # print(moving_img_Path)
                #         fixed_seg_path  = fixed_img_path[:-7] + "_seg.nii.gz"
                #         moving_seg_Path = moving_img_Path[:-7] + "_seg.nii.gz"
                #
                #         fixed_seg  = sitk.GetArrayFromImage(sitk.ReadImage(fixed_seg_path))
                #         fixed_seg  = np.swapaxes(fixed_seg,0,2)
                #         #fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())
                #
                #         moving_seg = sitk.GetArrayFromImage(sitk.ReadImage(moving_seg_Path))
                #
                #         #convert to binary classes
                #         fixed_seg[fixed_seg>0.0]     = 1.0
                #         moving_seg[moving_seg > 0.0] = 1.0
                #         # fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())
                #
                #         # print("len  len(np.unique(fixed_seg)) : ", len(np.unique(fixed_seg)))
                #         # print("len  len(np.unique(moving_seg)): ", len(np.unique(moving_seg)))
                #
                #         #moving_seg = load_4D(moving_seg_Path)
                #         moving_seg = moving_seg[np.newaxis,...]
                #         moving_seg_tensor = torch.from_numpy(moving_seg).float().to(device).unsqueeze(dim=0)
                #
                #         tX = tX.to(device).float()
                #         tY = tY.to(device).float()
                #         # assert not np.any(np.isnan(X.cpu().numpy()))
                #         # assert not np.any(np.isnan(Y.cpu().numpy()))
                #
                #         # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
                #         with torch.no_grad():
                #            disp_fld, tX_Y, tY_4x, tF_xy, tF_xy_lvl1, tF_xy_lvl2, _ = model(tX, tY)
                #         #disp_fld, tX_Y, tY_4x, tF_xy, tF_xy_lvl1, tF_xy_lvl2, _ = model.eval(tX, tY)
                #         transformed_seg_image = transform(moving_seg_tensor, disp_fld.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
                #         transformed_seg_image[transformed_seg_image > 0.0] = 1.0
                #
                #         # print("fixed_seg                :",fixed_seg.shape)
                #         # print("transformed_moving_image :",transformed_seg_image.shape)
                #         # print("len  len(np.unique(transformed_seg_image)): ", len(np.unique(transformed_seg_image)))
                #         # computet he dice
                #         gt = fixed_seg.ravel();
                #         res = transformed_seg_image.ravel()
                #         total_avg_dice = total_avg_dice + np.sum(res[gt == 1]) * 2.0 / (np.sum(res) + np.sum(gt))
                #         testCounter+=1
                #     total_avg_dice = total_avg_dice / testCounter


            if (lvlID==3) and (step == freeze_step):
                model.unfreeze_modellvl2()

            step += 1

            if step > num_iteration:
                break
        print("one epoch pass ....")

    model_lvl_path = model_dir + '/' +           model_name + "stagelvl" + str(lvlID) + "_" + str(num_iteration) + '.pth'
    loss_lvl_path  = model_dir + '/' + 'loss'  + model_name + "stagelvl" + str(lvlID) + "_" + str(num_iteration) + '.npy'
    torch.save(model.state_dict(), model_lvl_path)
    #np.save(loss_lvl_path, lossall)
    return model_lvl_path

# ................................... Start training  .....................................


model_lvl1_path = train(1,opt)
model_lvl2_path = train(2,opt, model_lvl1_path)
model_lvl3_path = train(3,opt, model_lvl1_path , model_lvl2_path)
