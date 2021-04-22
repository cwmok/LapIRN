import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools, os, time, shutil
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt

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


# def generate_grid_unit(imgshape):
#     x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
#     y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
#     z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
#     grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
#     grid = np.swapaxes(grid, 0, 2)
#     grid = np.swapaxes(grid, 1, 2)
#     return grid


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

# it seems this class is not used at all
# class Dataset(Data.Dataset):
#     'Characterizes a dataset for PyTorch'
#
#     def __init__(self, names, iterations, norm=True):
#         'Initialization'
#         self.names = names
#         self.norm = norm
#         self.iterations = iterations
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return self.iterations
#
#     def __getitem__(self, step):
#         'Generates one sample of data'
#         # Select sample
#         index_pair = np.random.permutation(len(self.names))[0:2]
#         img_A = load_4D(self.names[index_pair[0]])
#         img_B = load_4D(self.names[index_pair[1]])
#         if self.norm:
#             return Norm_Zscore(imgnorm(img_A)), Norm_Zscore(imgnorm(img_B))
#         else:
#             return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm  = norm
        self.index_pair = sorted(list(itertools.permutations(names, 2)))
        # print("len(names)            : ",len(names) )            # 25
        # print("len(self.index_pair ) : ", len(self.index_pair )) # 600:   comb(25,2)=300 * 2 = 600
        # print(self.index_pair[:3])
        # print(ok)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        #print("----------------Dataset_epoch __getitem__start------------------------")
        # Select sample
        pairIndex = step
        if step < len(self.index_pair):
            pairIndex = step - len(self.index_pair)
        img_A = load_4D(self.index_pair[pairIndex][0]) # moving image
        img_B = load_4D(self.index_pair[pairIndex][1]) # fixed image

        # it is important to normalise to 0 1 range to avoid negative loss
        # min_max_scaler = MinMaxScaler()
        # img_A =min_max_scaler.fit_transform(img_A)
        # img_B =min_max_scaler.fit_transform(img_B)
        # print("---------------- before  ------------------------")
        # print("img_A img_B. : ", len(np.unique(img_A)),len(np.unique(img_B)))
        # print("img_A.min(),img_A.max() : ", img_A.min(), img_A.max())
        # print("img_B.min(),img_B.max() : ", img_B.min(), img_B.max())
        # print("---------------- after ------------------------")
        # normalisation between 0 and 1
        # TODO: normalisation between -1 and 1
        img_A = (img_A - img_A.min()) / (img_A.max() - img_A.min())
        img_B = (img_B - img_B.min()) / (img_B.max() - img_B.min())
        # print("img_A img_B. : ", len(np.unique(img_A)),len(np.unique(img_B)))
        # print("img_A.min(),img_A.max() : ", img_A.min(), img_A.max())
        # print("img_B.min(),img_B.max() : ", img_B.min(), img_B.max())
        outputImages = torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()
        # print("---------------- after float ------------------------")
        # print("img_A img_B. : ", len(np.unique(img_A)),len(np.unique(img_B)))
        # print("img_A.min(),img_A.max() : ", img_A.min(), img_A.max())
        # print("img_B.min(),img_B.max() : ", img_B.min(), img_B.max())

        if self.norm:
            outputImages =Norm_Zscore(imgnorm(img_A)), Norm_Zscore(imgnorm(img_B))
        #print("----------------Dataset_epoch __getitem__end------------------------")
        return   [outputImages, self.index_pair[pairIndex]]

# it seems this class is not used at all
# class Predict_dataset(Data.Dataset):
#     def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True):
#         super(Predict_dataset, self).__init__()
#         self.fixed_list = fixed_list
#         self.move_list = move_list
#         self.fixed_label_list = fixed_label_list
#         self.move_label_list = move_label_list
#         self.norm = norm
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.move_list)
#
#     def __getitem__(self, index):
#         fixed_img = load_4D(self.fixed_list)
#         moved_img = load_4D(self.move_list[index])
#         fixed_label = load_4D(self.fixed_label_list)
#         moved_label = load_4D(self.move_label_list[index])
#
#         if self.norm:
#             fixed_img = Norm_Zscore(imgnorm(fixed_img))
#             moved_img = Norm_Zscore(imgnorm(moved_img))
#
#         fixed_img = torch.from_numpy(fixed_img)
#         moved_img = torch.from_numpy(moved_img)
#         fixed_label = torch.from_numpy(fixed_label)
#         moved_label = torch.from_numpy(moved_label)
#
#         if self.norm:
#             output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
#                       'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
#             return output
#         else:
#             output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
#                       'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
#             return output

def iaLog2Fig(logPath):
    # convert log file to figures:
    stepsLst = [];    lossLst = [];    simNCCLst = [];    JdetLst = [];    smoLst = [];    LossAll = []
    wdPath =  os.path.dirname(logPath)+'/'
    logName = logPath.split('/')[-1][:-4]
    figLossTrnPath  = wdPath + "lossTrn.png" ;    figLossSimPath  = wdPath + "lossSim.png"
    figLossJdetPath = wdPath + "lossJdet.png";    figLossSmoPath  = wdPath + "lossSmo.png"
    figLossAllPath  = wdPath + "lossAll.png"
    try:
        #print("reading file data : " + logPath)
        f = open(logPath,'r')
        lines =f.readlines()
        #print(len(lines))
        labels = ['steps','loss','sim_NCC','Jdet','smo']
        for line in lines:
            #print(len(line))
            if len(line)>1:
               #   0       1     2      3          4          5         6       7         8            9    10          11            12       13
               #['step', '"0"', '->', 'training', 'loss', '"-0.2498"', '-', 'sim_NCC', '"-0.250243"', '-', 'Jdet', '"0.0000000000"', '-smo', '"0.0005"']
               data = line.split()
               #print (data)
               step = int(  data[1].strip('"'))
               loss = float(data[5].strip('"'))
               sim_NCC = float(data[8].strip('"'))
               Jdet = float(data[11].strip('"'))
               smo = float(data[13].strip('"'))
               stepsLst.append( step ) ; lossLst.append(loss) ; simNCCLst.append(sim_NCC) ;  JdetLst.append(Jdet)
               smoLst.append(smo) ;   LossAll.append([step, loss,sim_NCC,Jdet,smo])

        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, lossLst      , label='Training Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossTrn.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, simNCCLst      , label='simNCC Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossSim.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        plt.plot(stepsLst, JdetLst      , label='Jdet Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossJdet.png')
        plt.clf() ;        plt.cla() ;            plt.close()

        plt.plot(stepsLst, smoLst      , label='Smoothing Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossSmo.png')
        plt.clf() ;        plt.cla() ;            plt.close()

        plt.plot(stepsLst, lossLst      , label='Training Loss') ;
        plt.plot(stepsLst, simNCCLst      , label='simNCC Loss') ;
        plt.plot(stepsLst, JdetLst      , label='Jdet Loss') ;
        plt.plot(stepsLst, smoLst      , label='Smoothing Loss') ;
        plt.legend() ;     plt.savefig(wdPath+logName+'_lossAll.png')
        plt.clf() ;        plt.cla() ;            plt.close()
        #print("figures are generated .............")

    except:
        print("error : file not found "+ logPath)

#get numpy array
def diceDistanceMetric(gt,res):
    gt= gt.ravel() ; res = res.ravel()
    # print(gt.shape, res.shape)
    # print(gt.shape, res.shape)
    # if len(np.unique(gt) ) !=2:
    #    gt[gt>=0.5]  = 1.0
    #    gt[gt < 1.0] = 0.0
    # if len(np.unique(res)) !=2:
    #    res[res>=0.5]  = 1.0
    #    res[res < 1.0] = 0.0
    # print(gt.min(),gt.max(),len(np.unique(gt)) )
    # print(res.min(), res.max(), len(np.unique(res)))
    iaDice = distance.dice(gt, res)
    return iaDice

def diceMetric(gt,res):
    gt= gt.ravel() ; res = res.ravel()
    # print("gt.shape   : ", gt.shape   )
    # print("res.shape  : ", res.shape)
    iaDice = distance.dice(gt, res)
    return iaDice
# def diceMetric(gt,res):
#     gt= gt.ravel() ; res = res.ravel()
#     print("gt.shape   : ", gt.shape   )
#     print("res.shape  : ", res.shape)
#     indxGt  = []#np.where(gt==1)
#     indxGt  = [x for x in gt if x==1 ]
#     indxRes = [x for x in res if x==1 ] #np.where(res==1)
#     s = 0.0
#     print("len(indxGt)  : ",len(indxGt)  )
#     print("len(indxRes) : ", len(indxRes))
#     print("indxGt: ",  indxGt[0])
#     print("indxRes: ", indxRes[0])
#     for x in indxGt:
#         for y in indxRes:
#             print("x: ", x)
#             print("y: ",y)
#             if x==y:
#                s+=1
#     print("s = ",s)
#     s= s/len(gt)
#     print("s = ",s)
#     iaDice =s
#     #iaDice = distance.dice(gt, res)
#     return iaDice

def mseMetric(gt,res):
    iaMSE = ( (gt[:] - res[:]) ** 2).mean()
    #iaMSE = distance.euclidean (gt, res)
    return iaMSE

#convert to binary image
def checkSeg(img):
    #img= img.ravel()
    if len(np.unique(img) ) !=2:
       img[img>=0.0]  = 1.0
       #img[img < 1.0] = 0.0
    return img


# testing the current model
def check_metric(step, model, transform, grid,testingLst):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    counter          = 0
    totalDiceW       = 0.0
    totalDiceT       = 0.0
    total_avg_dice_W = 0.0
    total_avg_dice_T = 0.0
    for i in range(len(testingLst)-1 ):
        counter += 1
        #get two images
        fixed_img_path     = testingLst[i]
        moving_img_Path    = testingLst[i+1]
        fixed_seg_path     = fixed_img_path [:-7]+"_seg.nii.gz"
        moving_seg_Path    = moving_img_Path[:-7]+"_seg.nii.gz"

        print(fixed_img_path, moving_img_Path)

        fixedName  = fixed_img_path.split('/')[-1][:-11]
        movingName = moving_img_Path.split('/')[-1][:-11]
        #print(fixedName, movingName)
        #print(ok)
        fixed_img  = load_4D(fixed_img_path)
        moving_img = load_4D(moving_img_Path)

        fixed_seg  = load_4D(fixed_seg_path)
        moving_seg = load_4D(moving_seg_Path)

        fixed_img_tensor  = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
        moving_img_tensor = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

        fixed_seg_tensor  = torch.from_numpy(fixed_seg).float().to(device).unsqueeze(dim=0)
        moving_seg_tensor = torch.from_numpy(moving_seg).float().to(device).unsqueeze(dim=0)


        with torch.no_grad():
            displacement_field, wrapped_moving_image_tensor, fixed_image_out_tensor, velocity_field, lvl1_v, lvl2_v, _= model(moving_img_tensor, fixed_img_tensor)
            transformed_moving_image = transform(moving_seg_tensor, displacement_field.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
            fixed_image_out          = fixed_seg_tensor.cpu().numpy()[0, 0, :, :, :]

            print("fixed_image_out          : ",len(np.unique(fixed_image_out)))
            print("transformed_moving_image : ",len(np.unique(transformed_moving_image)))
            print("transformed_moving_image : ",(np.unique(transformed_moving_image)))

            #print(ok)
            fixed_image_out          = checkSeg(fixed_image_out)
            transformed_moving_image = checkSeg(transformed_moving_image)
            #wrapped_moving_image     = checkSeg(wrapped_moving_image)

            print("fixed_image_out          : ",len(np.unique(fixed_image_out)))
            print("transformed_moving_image : ",len(np.unique(transformed_moving_image)))
            print("transformed_moving_image : ",(np.unique(transformed_moving_image)))

            #dicew = diceMetric(fixed_image_out, wrapped_moving_image)
            dicet = diceMetric(fixed_image_out, transformed_moving_image)
            print(dicet)
            dicew = diceMetric(fixed_image_out, fixed_image_out)
            dicet = diceMetric(transformed_moving_image, transformed_moving_image)
            print(dicew,dicet)

            print(ok)
            totalDiceT += dicet
            totalDiceW += dicew
            # if totalDiceW >0.9:
            #     shutil.copyfile(fixed_img_path, model_dir+'/'+'fixedImage.nrrd')
            #     save_img_3d(fixed_image_out         , fixed_img_path,  model_dir+'/'+fixedName+'_out'+ str(step)+'-label.nrrd')
            #     save_img_3d(transformed_moving_image, fixed_img_path,  model_dir+'/'+movingName+'_transformed'+ str(step)+'-label.nrrd')
    total_avg_dice_T    = totalDiceT/counter
    total_avg_dice_W    = totalDiceW/counter
    return total_avg_dice_W, total_avg_dice_T