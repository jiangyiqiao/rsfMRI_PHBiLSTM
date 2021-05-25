import math
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Bidirectional, LSTM, Flatten
import tensorflow as tf
from tensorflow import set_random_seed

np.random.seed(1)
set_random_seed(2)

class HBiLSTM(object):
    def __init__(self, x_train, y_train, x_test, y_test, n_classes, featureNum):
        self.n_classes = n_classes
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

    def run(self):
        learning_rate = 10e-4
        lambda_loss_amount = 0.0015
        droup_out = 0.9

        n_steps = 180
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
                                                                                    specificity, f_score, auc
                                                                                    ))
                    print(tn, fp, fn, tp)

                    return test_acc, sensitivity, specificity, f_score, auc
