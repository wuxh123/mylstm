# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import talib

class LstmPred(object):
    rnn_unit = 15  # 隐层神经元的个数
    lstm_layers = 3  # 隐层层数
    input_size = 9
    output_size = 1
    time_step = 8
    target_index = 9
    lr = 0.0005  # 学习率
    scale=0.8
    batch_size=32
    code=''
    train_begin = 0
    train_end = 0
    test_begin = 0
    test_end = 0

    def __init__(self):
        tf.reset_default_graph()
        self.weights = {
            'in': tf.Variable(tf.random.normal([self.input_size, self.rnn_unit])),
            'out': tf.Variable(tf.random.normal([self.rnn_unit, 1]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }
        with tf.device('/cpu:0'):
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            #

    # ——————————————————导入数据——————————————————————
    def openfile(self,file):
        #f = open(file)
        #df = pd.read_csv(f)  # 读入股票数据
        #self.data = df.iloc[:, 3:12].values  # 取第3-10列
        self.data = np.loadtxt(file, skiprows=1, delimiter=",", usecols=(3,4,5,6,7,9,10,11))
        #self.data = np.loadtxt(file, skiprows=1, delimiter=",", usecols=(3, 9, 10, 11))
        avg_5 = talib.MA(self.data[:,0], timeperiod=5)
        # avg_10 = talib.MA(self.data[:, 0], timeperiod=10)
        # avg_30 = talib.MA(self.data[:, 0], timeperiod=30)
        self.data=np.c_[self.data, avg_5]
        # self.data=np.c_[self.data, avg_10]
        # self.data=np.c_[self.data, avg_30]

        # macd = 12  天  EMA - 26  天
        # EMA （DIFF）signal = 9   天
        # MACD的EMA  （DEA）hist = MACD - MACD  signal （DIFF - DEA）
        #macd, macdsignal, macdhist = talib.MACD(self.data[:,0], fastperiod=12, slowperiod=26, signalperiod=9)
        #self.data = np.c_[self.data, macd]
        #self.data = np.c_[self.data, macdsignal]
        #self.data = np.c_[self.data, macdhist]

        #upperband, middleband, lowerband = talib.BBANDS(self.data[:,0], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        #self.data = np.c_[self.data, upperband]
        #self.data = np.c_[self.data, middleband]
        #self.data = np.c_[self.data, lowerband]

        # real1_ = talib.EMA(self.data[:,0], timeperiod=30)
        # self.data = np.c_[self.data, real1_]
        #
        # real2 = talib.SMA(self.data[:,0], timeperiod=30)
        # self.data = np.c_[self.data, real2]

        self.data=np.c_[self.data, self.data[:,1]]

        for i in range(0,5):
             self.data = np.delete(self.data,[0],axis=0)

        #a = talib.CDL2CROWS(self.data[:,0])

        #data最后一行空，分别对每行插入明日价格
        for v in range(0,len(self.data)-2):
            self.data[v][self.target_index] = self.data[v+1][5]
        self.data[len(self.data)-1][self.target_index] = 1e-8

        self.train_begin = len(self.data)-int((math.floor(len(self.data)*self.scale/self.time_step))*self.time_step)+self.time_step-1
        self.train_end = len(self.data)-1
        self.test_end=self.train_begin
        self.test_begin=self.test_end%self.time_step
        if self.test_end - self.test_begin < self.time_step:
            self.test_begin=self.test_end=0
        if len(self.data) > 200:
            self.batch_size = 32
        elif len(self.data)>100:
            self.batch_size=32
        elif len(self.data)>64:
            self.batch_size=16
        else:
            self.batch_size=4

    # 获取训练集
    def get_train_data(self):
        #train_begin = len(self.data)-int(process.floor(len(self.data)*self.scale/self.time_step)*self.time_step)
        #train_end = len(self.data)#int(process.ceil(len(self.data)*self.scale/self.time_step)*self.time_step)
        batch_index = []
        data_train = self.data[self.train_begin:self.train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data)):
            if i % self.batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + self.time_step, :self.input_size]
            y = normalized_train_data[i:i + self.time_step, self.target_index, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        #batch_index.append((len(normalized_train_data) - self.time_step))
        return batch_index, train_x, train_y


    # 获取测试集
    def get_test_data(self):
        # test_begin = int(process.ceil(len(self.data)*self.scale/self.time_step)*self.time_step)
        # test_end = process.floor((len(self.data)/self.time_step))*self.time_step
        data_test = self.data[self.test_begin:self.test_end]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std  # 标准化
        size = len(normalized_test_data) // self.time_step  # 有size个sample
        test_x, test_y = [], []
        for i in range(size):
            x = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, :self.input_size]
            y = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, self.target_index]
            test_x.append(x.tolist())
            test_y.extend(y)
        # test_x.append((normalized_test_data[(i + 1) * self.time_step:, :self.input_size]).tolist())
        # test_y.extend((normalized_test_data[(i + 1) * self.time_step:, self.target_index]).tolist())
        return mean, std, test_x, test_y

    # 获取测试集
    # def get_test_data1():
    #     time_step = 4
    #     test_begin = len(data)-int(len(data)*0.8)
    #     data_test = data[test_begin:]
    #     #mean = np.mean(data_test, axis=0)
    #     #std = np.std(data_test, axis=0)
    #     #normalized_test_data = (data_test - mean) / std  # 标准化
    #     size = (len(data_test) + time_step - 1) // time_step  # 有size个sample
    #     test_x, test_y = [], []
    #     for i in range(size - 1):
    #         x = data_test[i * time_step:(i + 1) * time_step, :9]
    #         y = data_test[i * time_step:(i + 1) * time_step, 6]
    #         test_x.append(x.tolist())
    #         test_y.extend(y)
    #     test_x.append((data_test[(i + 1) * time_step:, :9]).tolist())
    #     test_y.extend((data_test[(i + 1) * time_step:, 6]).tolist())
    #     return test_x, test_y

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置、dropout参数

    # ——————————————————定义神经网络变量——————————————————
    def lstmCell(self):
        # basicLstm单元
        basicLstm = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
        # dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=self.keep_prob)
        return basicLstm


    def lstm(self,X):
        with tf.device('/cpu:0'):
            lstm_batch_size = tf.shape(X)[0]
            lstm_time_step = tf.shape(X)[1]
            w_in = self.weights['in']
            b_in = self.biases['in']
            input = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, lstm_time_step,self. rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstmCell() for i in range(self.lstm_layers)])
            init_state = cell.zero_state(lstm_batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, self.rnn_unit])
            w_out = self.weights['out']
            b_out = self.biases['out']
            pred = tf.matmul(output, w_out) + b_out
            return pred, final_states


    # ————————————————训练模型————————————————————

    def train_lstm(self):
        with tf.device('/cpu:0'):
            X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
            Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])
            batch_index, train_x, train_y = self.get_train_data()
            with tf.variable_scope("sec_lstm"):
                pred, _ = self.lstm(X)
            loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(5000):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
                    for step in range(len(batch_index) - 1):
                        _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                         Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                         self.keep_prob: 0.5})
                    print("Number of iterations:", i, " loss:", loss_)
                print("model_save: ", saver.save(sess, self.code+'\\modle.ckpt'))
                # I run the code on windows 10,so use  'model_save2\\modle.ckpt'
                # if you run it on Linux,please use  'model_save2/modle.ckpt'
                print("The train has finished")

    # ————————————————预测模型————————————————————
    def prediction(self):
        X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
        mean, std, test_x, test_y = self.get_test_data()
        with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
            pred, _ = self.lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint(self.code)
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x)):
                prob = sess.run(pred, feed_dict={X: [test_x[step]], self.keep_prob: 1})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[self.target_index] + mean[self.target_index]
            test_predict = np.array(test_predict) * std[self.target_index] + mean[self.target_index]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / (test_y[:len(test_predict)]+1e-9))
            print("The accuracy of this predicttest_predict:", acc)

            # 以折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b', )
            plt.plot(list(range(len(test_y))), test_y, color='r')
            plt.show()

    #prediction()


    # ————————————————预测模型————————————————————
    # def prediction1(time_step=4):
    #     X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    #     test_x, test_y = get_test_data1()
    #     with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
    #         pred, _ = lstm(X)
    #     saver = tf.train.Saver(tf.global_variables())
    #     with tf.Session() as sess:
    #         # 参数恢复
    #         module_file = tf.train.latest_checkpoint('model_save2')
    #         saver.restore(sess, module_file)
    #         test_predict = []
    #         for step in range(len(test_x)):
    #             prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
    #             predict = prob.reshape((-1))
    #             test_predict.extend(predict)
    #         test_y = np.array(test_y)
    #         test_predict = np.array(test_predict)
    #         acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
    #         print("The accuracy of this predicttest_predict:", acc)
    #
    #         # 以折线图表示结果
    #         plt.figure()
    #         plt.plot(list(range(len(test_predict))), test_predict, color='b', )
    #         plt.plot(list(range(len(test_y))), test_y, color='r')
    #         plt.show()
    #
    # prediction1()

    # ————————————————预测模型————————————————————
    def predict1day(self,time_step=1):
        with tf.device('/cpu:0'):
            X = tf.placeholder(tf.float32, shape=[None, time_step, self.input_size])
            with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
                pred, _ = self.lstm(X)
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                #参数恢复
                module_file = tf.train.latest_checkpoint(self.code)
                saver.restore(sess, module_file)

                #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
                prev_seq=self.data[-1:,(0,1,2,3,4,5,6,7,8)]
                predict=[]
                #得到之后100个预测结果
                #for step in range(10):
                next_seq = sess.run(pred, feed_dict={X:[prev_seq], self.keep_prob: 1})
                predict.append(next_seq[-1])
                    #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                    #prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
                #以折线图表示结果
                # plt.figure()
                # plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
                # plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
                # plt.show()
                rtn= ((predict[0]/100)+1)*prev_seq[0][0]
                #rtn=next_seq[-1]+prev_seq[0][0]
                print(rtn)
                return rtn

    def predictonce(self,code):
        filename=code + ".csv"
        self.code=code
        self.openfile(filename)
        print("++++++++++++++++++++++++++++++++++++++")
        self.train_lstm()
        print("------------------------------------------")
        self.prediction()
        pred=self.predict1day()
        return pred

code='600649'
lstm = LstmPred()
pred = lstm.predictonce(code)