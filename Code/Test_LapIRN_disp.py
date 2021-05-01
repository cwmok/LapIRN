import os, shutil, time
from argparse import ArgumentParser

import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.data as Data
# from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
#     generate_grid, saveLog, iaLog2Fig, check_metric, load_4D,diceMetric,
from Functions import generate_grid, save_img, save_flow, transform_unit_flow_to_flow, load_4D,Dataset_epoch,diceMetric,normalizeAB
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
parser.add_argument("--datapath", type=str,
                    dest="datapath", default='../Data',
                    help="dataset path ")
parser.add_argument("--multiple_test", type=int,
                    dest="multiple_test", default=0,
                    help="test multiple images")

opt = parser.parse_args()
print(opt)

multiple_test = opt.multiple_test
datapath    = opt.datapath
savepath      = opt.savepath

fnms      = sorted(os.listdir(datapath))
names    = [os.path.join(datapath,x) for x in fnms if not "seg" in x]
namesSeg = [os.path.join(datapath,x) for x in fnms if     "seg" in x]
trainingLst = names[:-5]
testingLst  = names[-5:]
print((datapath))
print("len(trainingLst) : ",len(trainingLst))
print("len(testingLst)  : ",len(testingLst))

fixed_path    = opt.fixed
moving_path   = opt.moving

if multiple_test:
   fixed_path    = testingLst[0]
   moving_path   = testingLst[1]

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

# why is train true ?
model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(in_channel, n_classes, start_channel, is_train=isTrainLvl1,
                                                         imgshape=imgshape_4, range_flow=range_flow).cuda()
model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(in_channel, n_classes, start_channel, is_train=isTrainLvl2,
                                                         imgshape=imgshape_2, range_flow=range_flow,
                                                         model_lvl1=model_lvl1).cuda()
model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(in_channel, n_classes, start_channel, is_train=isTrainLvl3,
                                                    imgshape=imgshape, range_flow=range_flow,
                                                    model_lvl2=model_lvl2).cuda()

transform = SpatialTransform_unit().cuda()
model.load_state_dict(torch.load(opt.modelpath))
print("pretrained model is loaded: " + opt.modelpath)
model.eval()
transform.eval()

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print(" processor use: ", device)
grid = generate_grid(imgshape)
grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
#grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()


def testImages(testingImagesLst):
    totalDice = 0.0
    testing_generator = Data.DataLoader(Dataset_epoch(testingLst, norm=doNormalisation), batch_size=1, shuffle=True,  num_workers=0)
    for imgPair in testing_generator:
        moving_path = imgPair[1][1][0]
        fixed_path = imgPair[1][0][0]
        moving_tensor = imgPair[0][0]
        fixed_tensor  = imgPair[0][1]
        # print("moving_tensor.shape : ",moving_tensor.shape)
        # print("fixed_path  : ",fixed_path)
        # print("moving_path  : ", moving_path)
        dicePair = testOnePair(fixed_path,moving_path,  1  ,[moving_tensor,fixed_tensor])
        totalDice = totalDice +dicePair
    avgTotalDice = totalDice/len(imgPair)
    return avgTotalDice

def testOnePair(fixed_path,moving_path,saveResult=1,inputs=[]):
    #support nii, nii.gz, and nrrd
    ext = ".nii.gz" if ".nii.gz" in fixed_path else (".nii" if ".nii" in fixed_path else ".nrrd" )

    fixedName  = fixed_path.split('/')[-1][:-len(ext)]
    movingName = moving_path.split('/')[-1][:-len(ext)]

    if len(inputs)>0:
        print("input test images are loaded................")
        moving_img = inputs[0]
        fixed_img  = inputs[1]
        fixed_img  = fixed_img.float().to(device)
        moving_img = moving_img.float().to(device)
    else:
       print("loading input test images ................")
       fixed_img  = load_4D(fixed_path)
       moving_img = load_4D(moving_path)
       fixed_img  = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
       moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        #get displacement field

        displacement_field_tensor = model(moving_img, fixed_img)
        # convert to array for saving to file later
        displacement_field_tensor_cpu = displacement_field_tensor.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        displacement_field_array      = transform_unit_flow_to_flow(displacement_field_tensor_cpu)

        # get transformed image
        transformed_moving_array = transform(moving_img, displacement_field_tensor.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
        transformed_moving_array =  normalizeAB(transformed_moving_array,a=-500,b=800)

        print( "len unique transformed img : ", len(np.unique(transformed_moving_array) )  )
        print( "min max transformed img : ", np.min(transformed_moving_array) ,np.max(transformed_moving_array)    )

        print("transformed_moving_array     : ", transformed_moving_array.shape)

        # if segmentation found, transform segmentation as welll
        seg_path = moving_path[:-len(ext)]+'_seg'+ext
        if os.path.isfile(seg_path) and saveResult:
           seg_img =  load_4D(seg_path)
           seg_img = torch.from_numpy(seg_img).float().to(device).unsqueeze(dim=0)
           transformedSeg      = transform(seg_img, displacement_field_tensor.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
           print("len unique transformedSeg img : ", len(np.unique(transformedSeg)))
           print("min max transformedSeg img    : ", np.min(transformedSeg), np.max(transformedSeg))

           outputMovingSegPath = savepath + '/' + movingName + '_seg_warpped_moving'+ext
           save_img (transformedSeg              , outputMovingSegPath)




        # save result
        if saveResult:
            outputFixedPath     = savepath +'/'+ fixedName + ext
            outputDispFieldPath = savepath +'/'+ movingName  + '_warpped_flow'  + ext
            outputMovingPath    = savepath +'/'+ movingName  + '_warpped_moving'+ ext
            print("outputFixedPath     : " ,outputFixedPath )
            print("outputDispFieldPath : " ,outputDispFieldPath )
            print("outputMovingPath    : " ,outputMovingPath )

            shutil.copyfile(fixed_path , outputFixedPath)
            save_flow(displacement_field_array    , outputDispFieldPath)
            save_img (transformed_moving_array   , outputMovingPath)

        #evaluation:
        # compute dice
        fixedSeg = 0.0
        transformedMovingSeg = 0.0
        dicePair = 0.0

        #dicePair = diceMetric(fixedSeg,transformedMovingSeg)
        print(ok)
        return dicePair
    print("Finished")


if __name__ == '__main__':
    #imgshape = (160, 192, 144)
    if not opt.multiple_test:
       iaDice = testOnePair(fixed_path,moving_path)
    else:
       iaDice = testImages(testingLst)

