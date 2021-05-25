import argparse
import scipy.io
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default='PHBILSTM')
parser.add_argument("--data", type=str, default='kalman')
args = parser.parse_args()

from model.bilstm import BiLSTM
from model.hbilstm import HBiLSTM
from model.phbilstm import PHBiLSTM

if __name__ == '__main__':
    n_classes = 2
    if args.data == 'kalman':
        kalman = '0.1'
        Featurefile = 'data/features/kalmancorr_0.01_' + kalman + '_161.mat'
        datas = scipy.io.loadmat(Featurefile)
        corr = datas['datas']
    else:
        # SWC
        Featurefile = 'data/features/sliwinCorr_76.mat'  # 56 61 66 71 76 81
        datas = scipy.io.loadmat(Featurefile)
        corr = datas['sliwinData']
    sample_nums = len(corr)

    X = np.array([np.array(corr[i][0], dtype=np.float32) for i in range(sample_nums)])
    _, n_step, featureNum = X.shape
    print(X.shape)

    # 均值-标准差归一化具体公式是(x - mean)/std。
    # 其含义是：对每一列的数据减去这一列的均值，然后除以这一列数据的标准差。最终得到的数据都在0附近，方差为1。

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = preprocessing.scale(X[i][j])

    labels = np.array([0 if corr[i][3] == -1 else 1 for i in range(sample_nums)], dtype=np.int32)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    test_kflodCount = 1
    test_accs, sensitivitys, specificitys, f_scores, aucs = [], [], [], [], []
    for train_idx, test_idx in kf.split(X, labels):
        # 划分80%训练集，20% 测试集
        X_train = X[train_idx]
        X_test = X[test_idx]
        Y_train = labels[train_idx]
        Y_test = labels[test_idx]

        print('test_kflodCount', test_kflodCount)

        if args.model == 'BILSTM':
            networks = BiLSTM(X_train, Y_train, X_test, Y_test, n_classes, featureNum)
        elif args.model == 'HBILSTM':
            networks = HBiLSTM(X_train, Y_train, X_test, Y_test, n_classes, featureNum)
        elif args.model == 'PHBILSTM':
            networks = PHBiLSTM(X_train, Y_train, X_test, Y_test, n_classes, featureNum)
        else:
            assert "not correct model name, selected in BILSTM/HBILSTM/PHBILSTM"
        test_acc, sensitivity, specificity, f_score, auc = networks.run()

        test_accs.append(test_acc)
        sensitivitys.append(sensitivity)
        specificitys.append(specificity)
        f_scores.append(f_score)
        aucs.append(auc)

        print('test_kflodCount:{}'.format(test_kflodCount))

        test_kflodCount += 1
    print(
        'parameter of 5-fold mean of Test accuracy, sensitivity, specificity, f1_score, AUC_score:\n{}% {}% {}% {}% {}'.format(
            np.mean(test_accs), np.mean(sensitivitys), np.mean(specificitys), np.mean(f_scores),
            np.mean(aucs)))
