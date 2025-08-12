import numpy as np
from keras.src.utils import to_categorical
from numpy import matlib
import os
import cv2 as cv
import pandas as pd
from tqdm import tqdm
from BOA import BOA
from CFO import CFO
from DFA import DFA
from GRO import GRO
from Global_Vars import Global_Vars
from Model_GCN import Model_GCN
from Model_Inception import Model_Inception
from Model_MRA_GCNN_A_Bi_RNN import Model_MRA_GCNN_A_Bi_RNN
from Model_RNN import Model_RNN
from Objective_Function import objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *


def Read_Image(img_path):
    image = cv.imread(img_path)
    image = cv.resize(image, (512, 512))
    image = image.astype('uint8')
    return image


def Read_Dataset():
    Dataset = './Dataset/Dataset_1/'
    Datas_path = os.listdir(Dataset)
    Images = []
    Tar = []
    for n in range(len(Datas_path)):
        Class_dir = Dataset + Datas_path[n]
        if Datas_path[n] != 'Healthy':
            if os.path.isdir(Class_dir):
                Class_path = os.listdir(Class_dir)
                for i in range(len(Class_path)):
                    img_path = Class_dir + '/' + Class_path[i]
                    print(n, len(Datas_path), i, len(Class_path))
                    image = Read_Image(img_path)
                    name = img_path.split('/')[-2]
                    Tar.append(name)
                    Images.append(image)
    mapping = {'Mild DR': 0, 'Moderate DR': 1, 'Proliferate DR': 2, 'Severe DR': 3}
    Tar_encoded = np.array([mapping[t] for t in Tar])

    # Diabetic Macular Edema (Read image)
    Directory = './Dataset/messidor-2/messidor-2/preprocess/'
    path = './Dataset/messidor_data.csv'
    my_csv = np.asarray(pd.read_csv(path))
    Target_Column = my_csv[:, 2]
    Image_Column = my_csv[:, 0]
    Target_Class = []
    for k in tqdm(range(len(Target_Column))):
        if Target_Column[k] == 1:
            Target_Class.append(np.max(Tar_encoded + 1))
            filenames = Directory + Image_Column[k]
            Images.append(Read_Image(filenames))

    # Concatenating Targets
    Target = np.concatenate((Tar_encoded.reshape(-1, 1), (np.asarray(Target_Class).astype('int')).reshape(-1, 1)),
                            axis=0)
    Images = np.asarray(Images)
    class_tar = to_categorical(Target).astype('int')
    index = np.arange(len(Images))
    np.random.shuffle(index)
    Shuffled_Images = Images[index]
    Shuffled_Target = class_tar[index]
    return index, Shuffled_Images, Shuffled_Target


# Read Dataset
an = 0
if an == 1:
    Index, Images, Target = Read_Dataset()
    np.save('Index.npy', Index)
    np.save('Images.npy', Images)
    np.save('Target.npy', Target)

# Optimization for classification
an = 0
if an == 1:
    Feat = np.load('Images.npy', allow_pickle=True)  # Load the
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Target
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # hidden neuron count, Learning rate, Activation function Bi-RNN
    xmin = matlib.repmat(np.asarray([5, 0.01, 1]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 0.99, 5]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GRO...")
    [bestfit1, fitness1, bestsol1, time1] = GRO(initsol, fname, xmin, xmax, Max_iter)  # GRO

    print("DFA...")
    [bestfit2, fitness2, bestsol2, time2] = DFA(initsol, fname, xmin, xmax, Max_iter)  # DFA

    print("CFO...")
    [bestfit3, fitness3, bestsol3, time3] = CFO(initsol, fname, xmin, xmax, Max_iter)  # CFO

    print("BOA...")
    [bestfit4, fitness4, bestsol4, time4] = BOA(initsol, fname, xmin, xmax, Max_iter)  # BOA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_cls.npy', np.asarray(BestSol_CLS))  # Save the Best sol

# Classification using Batch size variation
an = 0
if an == 1:
    Feat = np.load('Images.npy', allow_pickle=True)  # Load the Images
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Targets
    BestSol = np.load('BestSol_cls.npy', allow_pickle=True)  # Load the BestSolution
    EVAL = []
    Batchsize = [4, 8, 16, 32, 64]
    for BS in range(len(Batchsize)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(BS, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred0 = Model_MRA_GCNN_A_Bi_RNN(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol,
                                                        BS=Batchsize[BS])
        Eval[5, :], pred1 = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[6, :], pred2 = Model_GCN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[7, :], pred3 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[8, :], pred4 = Model_MRA_GCNN_A_Bi_RNN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_BS.npy', np.asarray(EVAL))  # Save the Eval_all_SPE

plot_convergence()
ROC_curve()
Plot_Confusion()
Plot_batchsize()
Plot_KFold()
Sample_images()
