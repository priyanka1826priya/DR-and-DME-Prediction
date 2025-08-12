import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation
from Global_Vars import Global_Vars
from Model_MRA_GCNN_A_Bi_RNN import Model_MRA_GCNN_A_Bi_RNN


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_MRA_GCNN_A_Bi_RNN(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval = ClassificationEvaluation(pred, Test_Target)
            Fitn[i] = (1 / (Eval[4] + Eval[13]))  # 1 / (Accuracy + MCC)

        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_MRA_GCNN_A_Bi_RNN(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
        Eval = ClassificationEvaluation(pred, Test_Target)
        Fitn = (1 / (Eval[4]) + Eval[13])  # 1 / (Accuracy + MCC)
        return Fitn