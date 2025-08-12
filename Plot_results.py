from itertools import cycle
import numpy as np
import cv2 as cv
from prettytable import PrettyTable
from matplotlib import pylab
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")


def Statastical(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_convergence():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GRO-MRA-GNet+OBi-RNN', 'DFA-MRA-GNet+OBi-RNN', 'CFO-MRA-GNet+OBi-RNN', 'BOA-MRA-GNet+OBi-RNN', 'FBIDO-MRA-GNet+OBi-RNN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv[j, :] = Statastical(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv[j, :])
    print('-------------------------------------------------- Statistical Report ',
          '  --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#0cff0c', linewidth=3, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#aa23ff', linewidth=3, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.savefig("./Results/Convergence.png" )
    plt.show()


def ROC_curve():
    lw = 2
    cls = ['Inception', 'GCN', 'RNN', 'MRA-GCNN-A-Bi-RNN', 'PROPOSED']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(
        ["#fe2f4a", "#0165fc", "#fcb001", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
    Classes = ['Mild \nDR', 'Moderate \nDR', 'Proliferate \nDR', 'Severe \nDR', 'DME']
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(Classes)
    ax.yaxis.set_ticklabels(Classes)
    path = "./Results/Confusion.png"
    plt.savefig(path)
    plt.show()


def Plot_batchsize():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']

    Algorithm = ['GRO-MRA-GNet+OBi-RNN', 'DFA-MRA-GNet+OBi-RNN', 'CFO-MRA-GNet+OBi-RNN', 'BOA-MRA-GNet+OBi-RNN', 'FBIDO-MRA-GNet+OBi-RNN']
    Classifier = ['Inception-v3', 'GCN', 'RNN', 'MRA-GNet+Bi-RNN', 'FBIDO-MRA-GNet+OBi-RNN']
    Graph_Terms = [0, 1, 2, 3, 5, 7, 12]
    for j in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[j] + 4]
        Graph = Graph
        X = np.arange(Graph.shape[0])
        ax = plt.axes()
        colors = ['#ff000d', '#0cff0c', '#0652ff', '#e03fd8', 'black']
        plt.plot(X, Graph[:, 0], color='#ff000d', linewidth=4, marker='$\spadesuit$', markerfacecolor='#ffff81',
                 markersize=12, label=Algorithm[0])
        plt.plot(X, Graph[:, 1], color='#0cff0c', linewidth=4, marker='$\diamondsuit$', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(X, Graph[:, 2], color='#0652ff', linewidth=4, marker='$\clubsuit$', markerfacecolor='#bdf6fe',
                 markersize=12, label=Algorithm[2])
        plt.plot(X, Graph[:, 3], color='#e03fd8', linewidth=4, marker='$\U0001F601$', markerfacecolor='yellow',
                 markersize=12, label=Algorithm[3])
        plt.plot(X, Graph[:, 4], color='black', linewidth=4, marker='$\U00002660$', markerfacecolor='cyan',
                 markersize=12, label=Algorithm[4])
        plt.xticks(X, ('4', '8', '16', '32', '64'), fontsize=12)
        plt.grid(axis='y', linestyle='--', color='gray', which='major', alpha=0.8)
        plt.xlabel('Batch size', fontsize=13, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=13, fontweight='bold', color='#35530a')
        circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                          range(len(Algorithm))]
        ax.legend(circle_markers, Algorithm, title="", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  frameon=False, ncol=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        path = "./Results/Batch size_%s_line.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        Method_data = Graph[:, 5:]
        Batchsize = ['4', '8', '16', '32', '64']
        x = np.arange(Method_data.shape[0])
        width = 0.15
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#bf77f6']  # Colors for the regions

        for i, region in enumerate(Classifier):
            ax.bar(x + i * width, Method_data[:, i], width, label=region, color=colors[i])

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(Batchsize, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.xlabel('Batch size', fontsize=13, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=13, fontweight='bold', color='#35530a')
        circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                          range(len(Classifier))]
        ax.legend(circle_markers, Classifier, title="", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  frameon=False, ncol=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        path = "./Results/Batch size_%s_bar.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()



def Plot_KFold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Table_Term = [0, 1, 2, 4, 5, 7, 8, 9, 12]
    k_fold = ['1', '2', '3', '4', '5']

    Algorithm = ['TERMS', 'GRO-MRA-GNet+OBi-RNN', 'DFA-MRA-GNet+OBi-RNN', 'CFO-MRA-GNet+OBi-RNN', 'BOA-MRA-GNet+OBi-RNN', 'FBIDO-MRA-GNet+OBi-RNN']
    Classifier = ['TERMS', 'Inception-v3', 'GCN', 'RNN', 'MRA-GNet+Bi-RNN', 'FBIDO-MRA-GNet+OBi-RNN']
    for k in range(eval.shape[0]):
        value = eval[k, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of ',
              'Algorithm Comparison --------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of ',
              'Classifier Comparison --------------------------------------------------')
        print(Table)


def Sample_images():
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    if Target.shape[-1] >= 2:
        targ = np.argmax(Target, axis=1).reshape(-1, 1)
    else:
        targ = Target
    class_indices = {}
    for class_label in np.unique(targ):
        indices = np.where(targ == class_label)[0]
        class_indices[class_label] = indices
    for class_label, indices in class_indices.items():
        labels = ['Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR', 'DME']
        for i in range(5):
            print(labels[class_label], i + 1)
            Image = Images[indices[-i]]
            if len(Image.shape) == 2:
                Image = cv.cvtColor(Image, cv.COLOR_GRAY2BGR)
            cv.imshow('Image', Image)
            cv.waitKey(750)
            cv.imwrite(
                './Results/Sample_Images/' + str(labels[class_label]) + '_image_' + str(
                    i + 1) + '.png', Image)


if __name__ == '__main__':
    plot_convergence()
    ROC_curve()
    Plot_Confusion()
    Plot_batchsize()
    Plot_KFold()
    # Sample_images()
