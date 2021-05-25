import os
import scipy.io
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import math
import tensorflow as tf
from keras.layers import *
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)


class HABiLSTM(object):
    def __init__(self, x_train, y_train, x_test, y_test,  n_classes, n_step,featureNum):
        self.n_classes = n_classes
        self.n_step = n_step
        self.featureNum = featureNum
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = self.one_hot(y_train)
        self.y_test = self.one_hot(y_test)

    def one_hot(self, y_):
        # Function to encode neural one-hot output labels from number indexes
        # e.g.:
        # one_hot(y_=[[5], [0], [3]], n_classes=6):
        #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))
        return np.eye(self.n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    def hbilstm(self):
        learning_rate = 10e-4
        lambda_loss_amount = 0.0015
        droup_out = 0.9

        n_steps = self.n_step
        n_input = self.featureNum

        hidden_size = 64

        sess = tf.Session()
        K.set_session(sess)

        x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        # 全连接层
        fc_dense = Dense(hidden_size, kernel_initializer='uniform')(x)
        fc_dense = Dropout(droup_out)(fc_dense)
        fc_batchnorm = BatchNormalization()(fc_dense)
        fc_out = Activation('relu')(fc_batchnorm)

        bilstm_1 = Bidirectional(
            LSTM(hidden_size, return_sequences=True))(fc_out)
        bilstm_2 = Bidirectional(
            LSTM(hidden_size, return_sequences=True))(bilstm_1)

        bi_out = Flatten()(bilstm_2)  # 特征扁平化
        # 最后一层全连接层
        dense_out = Dense(hidden_size)(bi_out)
        dense_out = Dropout(droup_out)(dense_out)
        batchnorm = BatchNormalization()(dense_out)
        out = Activation('relu')(batchnorm)
        pred = Dense(self.n_classes, activation='softmax')(out)

        # 防止过拟合
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in
                                      tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + l2  # Softmax loss

        # 学习率衰减
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decay_step = 100
        decay_rate = 0.95
        current_epoch = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(learning_rate, global_step=current_epoch, decay_steps=decay_step,
                                        decay_rate=decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)  # Adam Optimizer

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Launch the graph
        # saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        sess.run(init)

        epochs = 1001
        train_losses = [3]

        for epoch in range(epochs):

            # 学习率衰减

            _, train_loss, acc, _ = sess.run(
                [optimizer, cost, accuracy, lr],
                feed_dict={
                    x: self.x_train,
                    y: self.y_train,
                    current_epoch: epoch,
                }
            )
            if epoch >= 100 and epoch % 100 == 0 or math.isnan(train_loss):
                train_losses.append(train_loss)
                acc = 100 * acc
                print("epoch: ", epoch)
                print("Training Accuracy = {}%".format(acc))
                print("Training Loss = {}".format(train_loss))
                if epoch == epochs - 1 or (
                        acc == 100 and np.mean(train_losses[-2]) - np.mean(train_losses[-1]) <= 0.05) or math.isnan(
                        train_loss):
                    test_acc, props = sess.run([accuracy, pred], feed_dict={x: self.x_test, y: self.y_test, })
                    # 计算其他指标
                    test_acc = 100 * test_acc
                    tn, fp, fn, tp = confusion_matrix(self.y_test.argmax(axis=1), props.argmax(axis=1)).ravel()
                    sensitivity = 100 * tp / (tp + fn)  # 敏感度
                    specificity = 100 * tn / (tn + fp)  # 特异性
                    f_score = 100 * 2 * tp / (2 * tp + fp + fn)
                    auc = metrics.roc_auc_score(self.y_test, props, average="weighted")
                    #
                    print("Test ACC,SEN,SPE,Fscore,AUC:\n{}% {}% {}% {}% {}".format(test_acc,
                                                                                     sensitivity,
                                                                                     specificity,f_score,auc
                                                                                     ))

                    print(tn, fp, fn, tp)
                    return test_acc, sensitivity,specificity,f_score,auc





if __name__ == '__main__':
    n_classes = 2
    Featurefile = 'data/features/sliwinCorr_81.mat'   # 56 61 66 71 76 81

    print(os.path.join(Featurefile))

    datas = scipy.io.loadmat(Featurefile)
    corr = datas['sliwinData']
    sample_nums = len(corr)

    X = np.array([np.array(corr[i][0], dtype=np.float32) for i in range(sample_nums)])
    _,n_step,featureNum = X.shape
    print(X.shape)

    # 均值-标准差归一化具体公式是(x - mean)/std。
    # 其含义是：对每一列的数据减去这一列的均值，然后除以这一列数据的标准差。最终得到的数据都在0附近，方差为1。

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = preprocessing.scale(X[i][j])

    labels = np.array([0 if corr[i][3] == -1 else 1for i in range(sample_nums)],dtype=np.int32)

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

        # if test_kflodCount < 3:
        #     test_kflodCount += 1
        #     continue
        #

        networks = HABiLSTM(X_train, Y_train, X_test, Y_test, n_classes,n_step, featureNum)
        test_acc, sensitivity, specificity, f_score, auc = networks.hbilstm()

        test_accs.append(test_acc)
        sensitivitys.append(sensitivity)
        specificitys.append(specificity)
        f_scores.append(f_score)
        aucs.append(auc)


        test_kflodCount += 1
    print(
        'Sliding Window 5-fold mean of Test accuracy, sensitivity, specificity, f1_score, AUC_score:\n{}% {}% {}% {}% {}'.format(np.mean(test_accs), np.mean(sensitivitys), np.mean(specificitys), np.mean(f_scores),np.mean(aucs)))


