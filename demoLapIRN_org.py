# -*- coding: utf-8 -*-
# =====================================================================

# @authors: Ibraheem Al-Dhamari, Jing, and Atul




# Unsupervised Deep Learning Registration for 3D Multi-modal Image

# sources:
#       https://github.com/cwmok/LapIRN
#       https://github.com/zhangjun001/ICNet/
#  to run testing on different GPU:
#
# last edit 14.4.2021

# =====================================================================
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Setup                                        ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# note:
#
# sudo pip3 install tensorflow==1.14 tensorflow-gpu==1.14 keras==2.3.1
# sudo pip3 install nibabel tqdm
'''
TODOS:
      - add option to use list of filenames
      - add evaluation e.g. dice    
'''
# all installation and import
# main imports
import os, sys, time, csv
# third party imports
import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt

# deep learning
import tensorflow as tf
import keras.backend as K
import keras.layers
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# example of horizontal shift image augmentation
from numpy import expand_dims
from PIL import Image

import glob
import itertools

# this line solve memory problem
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # first gpu


print("user arguments: ", sys.argv)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Parameters Setting                           ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(sys.argv)
# in colab, we only call this file
isLocal = 1
nb_gpus = 1  # len( get_available_gpus())

doTrain = 1
doTest  = not doTrain
# 3 multi-resolutions  lvl1 lvl2 lvl3
slvl1 = 0  # start epoch for level1
slvl2 = 0   # start epoch for level2
slvl3 = 0   # start epoch for level3
lvl1  = 3000  # 30001 # number of iterations for level1
lvl2  = 3000  # 30001 # number of iterations for level2
lvl3  = 3000  # 60001 # number of iterations for level3
checkpoint = 100  # 1000 #100
wd_path      = os.path.join( os.path.expanduser("~") , "myGitLab/DNN_ImageRegistration/IA/LapIRN_org/")
dataset_path = "/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/L2RMiccai2020/L2R_Task3_AbdominalCT_160x192x144"
step_size  = 1e-4 # 1e-4

if len(sys.argv) > 1:
    print("using user arguments .................")
    doTrain = int(sys.argv[1])
    doTest  = not doTrain
    isLocal = int(sys.argv[2])

#from iaUtils import *

print(tf.__version__)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Train                                        ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
if doTrain:
    # 2000x300 9.4 h
    print("--------------------------------------------")
    print("        Start Training                      ")
    print("--------------------------------------------")


    if len(sys.argv) > 1:
        print("using user arguments .................")
        # 3 multi-resolutions  lvl1 lvl2 lvl3
        slvl1 = int(sys.argv[3])
        slvl2 = int(sys.argv[4])
        slvl3 = int(sys.argv[5])
        lvl1  = int(sys.argv[6])
        lvl2  = int(sys.argv[7])
        lvl3  = int(sys.argv[8])
        checkpoint = int(sys.argv[9])
        wd_path = (sys.argv[10])
        dataset_path = (sys.argv[11])
    else:
        print("using default arguments .................")

    if isLocal:
        print("script is running in locally...........")
        # gitlab paths
        wd_path = os.path.join(os.path.expanduser("~"), "myGitLab/DNN_ImageRegistration/IA/LapIRN_org/")
    else:
        print("script is running in colab...........")

    sys.path.append(wd_path + 'LapIRN/Code')
    os.chdir(wd_path + 'LapIRN/Code')

    trainTime = time.time()
    #dataset_path = '/content/drive/MyDrive/DNN_Atul/datasets/LAPIRN datasets/Training/Scans'
    model_dir = ''
    trainDisp = 1
    training_args =  " --datapath "       + dataset_path  + \
                     " --lr "      + str(step_size)       + \
                     " --sIteration_lvl1 " + str(slvl1)   + \
                     " --sIteration_lvl2 " + str(slvl2)   + \
                     " --sIteration_lvl3 " + str(slvl3)   + \
                     " --iteration_lvl1 "  + str(lvl1)    + \
                     " --iteration_lvl2 "  + str(lvl2)    + \
                     " --iteration_lvl3 "  + str(lvl3)    + \
                     " --checkpoint "     + str(checkpoint)

    cmd =     "python3    Train_LapIRN_disp.py " + training_args
    print(cmd)
    if not trainDisp:
        cmd = "python3    Train_LapIRN_diff.py "  + training_args
    os.system(cmd)
    print("Training Time: ", time.time() - trainTime)
else:
    print("training is disabled")

if doTest:
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("                Test                                          ")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # test take fixed image, moving image, and model path then produces displacement field and transformed image
    lvl3 = 9000
    if len(sys.argv) > 1:
        print("using user arguments .................")
        # 3 multi-resolutions  lvl1 lvl2 lvl3
        lvl3         = int(sys.argv[3])
        wd_path      = sys.argv[4]
        dataset_path = sys.argv[5]
    else:
        print("using default arguments .................")

    if isLocal:
        print("script is running in locally...........")
        # gitlab paths
        wd_path = os.path.join(os.path.expanduser("~"), "myGitLab/DNN_ImageRegistration/IA/LapIRN_org/")
        # dataset_path = '/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/'+datasetName+'/'
        dataset_path = "/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/L2RMiccai2020/L2R_Task3_AbdominalCT_160x192x144"
    else:
        print("script is running in colab ...........")

    sys.path.append(wd_path + 'LapIRN/Code')
    os.chdir(wd_path + 'LapIRN/Code')

    testTime= time.time()
    #last model from training
    savepath  = '../Result'
    start_channel = ' 7 '

    # modelpath = '../Model/Stage/LDR_OASIS_NCC_unit_disp_add_reg_1_stagelvl3_' + str(lvl3) + '.pth'
    # fnms = sorted(os.listdir(dataset_path))
    # imgs = [x for x in fnms if  not "seg" in x ]
    # fixed_path    = os.path.join(dataset_path,imgs[0] )
    # moving_path   = os.path.join(dataset_path,imgs[1] )
    ## use default resized data
    ## fixed_path    = '../Data/image_A_160x192x144.nii.gz'
    ## moving_path   = '../Data/image_B_160x192x144.nii.gz'

    # test using default model and data, note the size change:
    modelpath = '../Model/LapIRN_disp_fea7.pth'
    fixed_path    = '../Data/image_A.nii'
    moving_path   = '../Data/image_B.nii'

    print("fixed_path  : ", fixed_path)
    print("moving_path : ", moving_path)
    testing_args  =  " --modelpath "       + modelpath     + \
                     " --savepath "        + savepath      + \
                     " --start_channel "   + start_channel + \
                     " --fixed  "          + fixed_path    + \
                     " --moving "          + moving_path


    cmd = "python3    Test_LapIRN_disp.py " + testing_args
    print(cmd)
    os.system(cmd)
    print("Testing Time: ", time.time() - testTime)

else:
    print("testing is disabled ! ................")
