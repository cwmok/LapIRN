# -*- coding: utf-8 -*-
# =====================================================================

# @authors: Ibraheem Al-Dhamari, Jing, and Atul




# Unsupervised Deep Learning Registration for 3D Multi-modal Image

# sources:
#       https://github.com/cwmok/LapIRN
#       https://github.com/zhangjun001/ICNet/
#  to run testing on different GPU:
#
# last edit 13.4.2021

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
   - test NCC vs MSE
   - create files for lists: training, testing 
   - check why training log is wrong.
   - check effect of opitmiser and step-size 
   - check if we can use dice loss
      
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # first gpu


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
lvl1 = 3   # 30001 # epochs/3
lvl2 = 3   # 30001 # epochs/3
lvl3 = 200 # 60001  # epochs/3
checkpoint = 100  # 1000 #100
wd_path      = os.path.join( os.path.expanduser("~") , "myGitLab/DNN_ImageRegistration/IA/LapIRN_org/")
dataset_path = "/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/L2RMiccai2020/L2R_Task3_AbdominalCT_160x192x144"

if len(sys.argv)>1:
    print("using user arguments .................")
    doTrain      = int(sys.argv[1])
    doTest     = not doTrain
    # 3 multi-resolutions  lvl1 lvl2 lvl3
    lvl1         = int(sys.argv[2])
    lvl2         = int(sys.argv[3])
    lvl3         = int(sys.argv[4])
    checkpoint   = int(sys.argv[5])
    wd_path      = (sys.argv[6])
    dataset_path = (sys.argv[7])
step_size  = 1e-4 # 1e-4


sys.path.append(wd_path + 'LapIRN/Code')
os.chdir(wd_path + 'LapIRN/Code')

if not isLocal:
    print("script is running in colab...........")
    #gitlab paths
    wd_path = os.path.join( os.path.expanduser("~") , "myGitLab/DNN_ImageRegistration/IA/LapIRN_org/")
    # dataset_path = '/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/'+datasetName+'/'
    dataset_path = "/mnt/hd8tb/ia_datasets/dnnDatasets/learn2reg_datasets/L2RMiccai2020/L2R_Task3_AbdominalCT"
    sys.path.append(wd_path + 'LapIRN/Code')
    os.chdir(wd_path + 'LapIRN/Code')
else:
    print("script is running locally...........")

#from iaUtils import *

print(tf.__version__)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Parameters Setting                           ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")



print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Dataset                           ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")



print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("                 Train                                        ")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
if doTrain:
    # 2000x300 9.4 h
    print("--------------------------------------------")
    print("        Start Training                      ")
    print("--------------------------------------------")
    trainTime = time.time()
    #dataset_path = '/content/drive/MyDrive/DNN_Atul/datasets/LAPIRN datasets/Training/Scans'
    model_dir = ''
    trainDisp = 1
    training_args =  " --datapath "       + dataset_path  +\
                     " --lr "      + str(step_size) + \
                     " --iteration_lvl1 " + str(lvl1)  + \
                     " --iteration_lvl2 " + str(lvl2) + \
                     " --iteration_lvl3 " + str(lvl3) + \
                     " --lr   "           + str(step_size) #+  \

    cmd =     "python3    Train_LapIRN_disp.py " + training_args

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
    testTime= time.time()
    #lvl3 = 25600
    j = lvl3 # test only the last model
    testLogFilePath =  '../Result/testLog.txt'
    testLogFile = open(testLogFilePath, "w+")
    logheadLine = "idx \t dice \r\n"
    testLogFile.write(logheadLine);
    testLogFile.close()
    while j<=lvl3:
        resultFile = open(testLogFilePath, "a+")
        resultFile.write(str(j) + "\t")
        resultFile.close()
        modelpath = '../Model/Stage/LDR_OASIS_NCC_unit_disp_add_reg_1_stagelvl3_' + str(j) + '.pth'
        testing_args =  " --modelpath "   + modelpath


        cmd = "python3    Test_LapIRN_disp.py " + testing_args
        print(cmd)
        os.system(cmd)
        j=j+checkpoint
    print("Testing Time: ", time.time() - testTime)

else:
    print("testing is disabled ! ................")