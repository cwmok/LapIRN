import itertools, os, time, shutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize


import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data
from torchvision import transforms
import torchio as tio
'''
TODOS:

add metrics: mse, dice 

'''

# generate 4d matrix (x,y,z,3) contains from -1 to 1 or from 0 to image max size
def generate_grid(imgshape, isUnit=0):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    if isUnit:
        x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
        y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
        z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    # create a grid then modify the axes
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    # modify the axes
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * z
    flow[:, :, :, 1] = flow[:, :, :, 1] * y
    flow[:, :, :, 2] = flow[:, :, :, 2] * x

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * z
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * y
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * x

    return flow

#inputs a medical image outputs a tensor
def img2tens(imgPath, isSeg=0,doNormalisation=1):
    #read an image, convert to array, normalise to 0,1, convert to tensor
    img = sitk.ReadImage(imgPath)
    print("img.GetSize() : ",img.GetSize() )
    imgA=sitk.GetArrayFromImage(img)
    print("imgA.shape : ", imgA.shape)
    if doNormalisation and not isSeg:
       imgA = (imgA - imgA.min()) / (imgA.max() - imgA.min())
       imgA[imgA>=0.5] = 1.0
       imgA[imgA<0.5] = 0.0
    img_tens = imgA[np.newaxis, ...]
    print(ok) # still need testing ....
    return   img_tens

#inputs a tensor, outputs a medical image
def tesn2img(img_tens, img_ref, img_output_path):
    if isinstance(img_ref,str):
       img_ref = sitk.ReadImage(img_ref)
    # get image info: size, spacing, origin, directions, datatype
    # convert from tensor to array
    # scale and convert type if needed
    # convert from array to image
    # if img_output_path is not none save the image
    img = 0
    print(ok) # still need testing
    return   img

def load_4D(name):
    # X = nib.load(name)
    # X = X.get_fdata()
    # X = np.reshape(X, (1,) + X.shape)
    X0 = nib.load(name)                  # image               e.g.   160, 192, 144
    X1 = X0.get_fdata()                  # nd array            e.g.   160, 192, 144
    X2 = np.reshape(X1, (1,) + X1.shape) # tensor (1,img_size) e.g. 1,160, 192, 144
    # print("X0.shape : ",X0.shape, type(X0))
    # print("X1.shape : ",X1.shape, type(X1))
    # print("X2.shape : ",X2.shape, type(X2))
    # print(ok)
    return X2



def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(N_I, index1=0.0001, index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def Norm_Zscore(img):
    img = (img - np.mean(img)) / np.std(img)
    return img


def save_img(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)

def saveLog(logPath,logLine):
    pass

def save_img_3d(imgA,refPath,outputPath):
    refImage = sitk.ReadImage(refPath)
    spc = refImage.GetSpacing() ;    org = refImage.GetOrigin();    dirs=  refImage.GetDirection()
    outImg = sitk.GetImageFromArray(imgA)
    outImg.SetSpacing(spc) ; outImg.SetOrigin(org) ; outImg.SetDirection(dirs)
    sitk.WriteImage(outImg,outputPath)

def save_flow_3d(imgA,refPath,outputPath):
    refImage = sitk.ReadImage(refPath)
    spc = refImage.GetSpacing() ;    org = refImage.GetOrigin();    dirs=  refImage.GetDirection()
    outImg = sitk.GetImageFromArray(imgA)
    outImg.SetSpacing(spc) ; outImg.SetOrigin(org) ; outImg.SetDirection(dirs)
    dft = sitk.DisplacementFieldTransform(outImg)
    #sitk.WriteImage(outImg, outputPath)
    sitk.WriteTransform(dft,outputPath)

def normalizeAB(x,a=0,b=1):
    #x = (x - x.min()) / (x.max() - x.min())
    x = (x - x.min()) / (x.max() - x.min()) * (b-a) + a
    return x

def img2SegTensor(imgPath,ext,d):
    seg_path = imgPath[:-len(ext)] + '_seg' + ext
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    seg[seg > 0] = 1.0;
    seg = np.swapaxes(seg, 0, 2)
    if d >1:
       seg = resize(seg, [int(x / d) for x in seg.shape], order=0)
    return seg

class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=1, aug=1, isSeg=0 , new_size=[0,0,0]):
        'Initialization'
        self.names = names
        self.norm  = norm
        self.index_pair = sorted(list(itertools.permutations(names, 2)))
        self.aug = aug
        self.isSeg = isSeg
        self.new_size = new_size

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        #print("----------------Dataset_epoch __getitem__start------------------------")
        pairIndex = step
        if step < len(self.index_pair):
            pairIndex = step - len(self.index_pair)
        moving_image_path = self.index_pair[pairIndex][0]
        fixed_image_path  = self.index_pair[pairIndex][1]
        # if self.isSeg:
        #     ext = ".nii.gz" if ".nii.gz" in fixed_image_path else (".nii" if ".nii" in fixed_image_path else ".nrrd")
        #     moving_image_path = moving_image_path[:-len(ext)]+"_seg"+ext
        #     fixed_image_path  = fixed_image_path[:-len(ext)]+"_seg"+ext

        movingImg = sitk.ReadImage(moving_image_path)
        fixedImg  = sitk.ReadImage(fixed_image_path)

        movingImgArray = sitk.GetArrayFromImage(movingImg) ; movingImgArray = np.swapaxes(movingImgArray,0,2).astype(np.float32)
        fixedImgArray  = sitk.GetArrayFromImage(fixedImg) ;  fixedImgArray  = np.swapaxes(fixedImgArray,0,2).astype(np.float32)
        # img_A = load_4D(moving_image_path) # moving image
        # img_B = load_4D(fixed_image_path) # fixed image
        if self.isSeg:
            movingImgArray [movingImgArray>0]=1.0
            fixedImgArray  [fixedImgArray>0] =1.0

        if self.new_size[0]>0:
            current_order = 0 if isSeg else 3
            movingImgArray = resize(movingImgArray, self.new_size, order=current_order)
            fixedImgArray  = resize(fixedImgArray,  self.new_size, order=current_order)

        movingImgTensor = movingImgArray[np.newaxis,...]
        fixedImgTensor  = fixedImgArray[np.newaxis,...]

        #print("movingImgTensor type : ", type(movingImgTensor))
        if self.aug:
            sz = movingImg.GetSize() ; szX = sz[0] ; szY = sz[1] ; szZ = sz[2] ;
            aug_probability = 0.1
            scalingPars         = (0.2*szX,1.2*szX,0.2*szY,1.2*szY,0.2*szZ,1.2*szZ)
            rotation_degrees    = (-10, 10)
            translatingPars     = (-0.05*szX,0.05*szX,-0.05*szY,0.05*szY,-0.05*szZ,0.05*szZ)
            transform = tio.RandomAffine(scalingPars,rotation_degrees,translatingPars)

            movingImgTensor = transform(movingImgTensor)
            fixedImgTensor  = transform(fixedImgTensor)

        # it is important to normalise to 0 1 range to avoid negative loss
        if not self.isSeg:
           movingImgTensor = normalizeAB(movingImgTensor,-500,800); movingImgTensor = normalizeAB(movingImgTensor) ;
           fixedImgTensor  = normalizeAB(fixedImgTensor,-500,800);   fixedImgTensor = normalizeAB(fixedImgTensor);

        outputImages = torch.from_numpy(movingImgTensor).float(), torch.from_numpy(fixedImgTensor).float()

        if self.norm:
            outputImages =Norm_Zscore(imgnorm(imgMA)), Norm_Zscore(imgnorm(imgFA))
        #print("----------------Dataset_epoch __getitem__end------------------------")
        return   [outputImages, self.index_pair[pairIndex]]



#convert to binary image
def checkSeg(img):
    #img= img.ravel()
    if len(np.unique(img) ) !=2:
       img[img>=0.0]  = 1.0
       #img[img < 1.0] = 0.0
    return img


def iaLog2Fig(logPath):
    # convert log file to figures:
    stepsLst = [];    lossLst = [];    simNCCLst = [];    JdetLst = [];    smoLst = [];    LossAll = []; diceLst=[]
    wdPath =  os.path.dirname(logPath)+'/'
    logName = logPath.split('/')[-1][:-4]
    figLossTrnPath  = wdPath + "lossTrn.png" ;    figLossSimPath  = wdPath + "lossSim.png"
    figLossJdetPath = wdPath + "lossJdet.png";    figLossSmoPath  = wdPath + "lossSmo.png"
    figLossAllPath  = wdPath + "lossAll.png"
    #try:
    if True:
        #print("reading file data : " + logPath)
        f = open(logPath,'r')
        lines =f.readlines()
        #print(len(lines))
        labels = ['steps','loss','sim_NCC','Jdet','smo','dice']
        for line in lines:
            #print(len(line))
            if len(line)>1:
               #   0       1     2      3          4          5         6       7         8            9    10          11            12       13           14          15
               #['step', '"0"', '->', 'training', 'loss', '"-0.2498"', '-', 'sim_NCC', '"-0.250243"', '-', 'Jdet', '"0.0000000000"', '-smo', '"0.0005"',  '-dice', '"0.0085"' ]
               data = line.split()
               #print (data)
               step = int(  data[1].strip('"'))
               loss = float(data[5].strip('"'))
               sim_NCC = float(data[8].strip('"'))
               Jdet = float(data[11].strip('"'))
               smo = float(data[13].strip('"'))
               dice = 0
               # print((data))
               # print(len(data))
               if len(data)>14:
                  dice  = float(data[15].strip('"'))
               stepsLst.append( step ) ; lossLst.append(loss) ; simNCCLst.append(sim_NCC) ;  JdetLst.append(Jdet)
               smoLst.append(smo) ;   LossAll.append([step, loss,sim_NCC,Jdet,smo,dice]) ; diceLst.append(dice)

        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, lossLst      , label='Training Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossTrn.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, simNCCLst      , label='Sim Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossSim.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, JdetLst      , label='Jdet Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossJdet.png')
        plt.clf() ;        plt.cla() ;            plt.close()

        plt.plot(stepsLst, smoLst      , label='Smoothing Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossSmo.png')
        plt.clf() ;        plt.cla() ;            plt.close()

        plt.plot(stepsLst, diceLst      , label='Dice metric') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_dice.png')
        plt.clf() ;        plt.cla() ;            plt.close()

        plt.plot(stepsLst, lossLst      , label='Training Loss') ;
        plt.plot(stepsLst, simNCCLst      , label='Sim Loss') ;
        plt.plot(stepsLst, JdetLst      , label='Jdet Loss') ;
        plt.plot(stepsLst, smoLst      , label='Smoothing Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossAll.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        #print("figures are generated .............")

    #except:
    #    print("error : file not found "+ logPath)



