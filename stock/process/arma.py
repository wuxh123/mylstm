# -*- coding: utf-8 -*-

# 本代码主要思路是利用ARIMA算法做时间序列预测
# 预测目标是2019年5月15日A股闭市时招商银行600036的股价
# 考虑到影响股价的因素的复杂性，以及金融投资的反身性理论，本次预测只使用了close的时间序列。更多的数据并没有什么用。

# 导入必须的模块
import tushare as ts  #使用的公开的数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts
import datetime

class clsArma(object):
    def __init__(self,code):
        # 导入数据，从去年1月开始即可
        endtime = str(datetime.datetime.now().date()).replace('-', '')
        self.data = ts.get_hist_data(code, start='2019-01-01', end=endtime).sort_index()

        # 数据安全和源数据备份
        #data.to_excel('600036.xlsx')


        # 只取 close 字段作为训练数据
        self.train = self.data['close']
        self.train.index = pd.to_datetime(self.train.index)  # 将字符串索引转换成时间索引
        self.train.tail()
        self.train.tail().index


    #使用ADF单位根检验法检验时间序列的稳定性
    #先做一个编译器, 让 adfuller 的输出结果更易读
    def tagADF(self,t):
        result = pd.DataFrame(index=["Test Statistic Value", "p-value", "Lags Used",
                                     "Number of Observations Used",
                                     "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                              columns=['value']
        )
        result['value']['Test Statistic Value']=t[0]
        result['value']['p-value']=t[1]
        result['value']['Lags Used']=t[2]
        result['value']['Number of Observations Used'] = t[3]
        result['value']['Critical Value(1%)']=t[4]['1%']
        result['value']['Critical Value(5%)']=t[4]['5%']
        result['value']['Critical Value(10%)']=t[4]['10%']
        print('t is:', t)
        return result

    def predict(self):
        # adfuller：全称 Augmented Dickey–Fuller test， 即扩展迪基-福勒检验，用来测试平稳性
        # adfuller检验是检查时间序列平稳性的统计测试之一。 这里的零假设是序列 train 是非平稳的。
        # 测试结果包括测试统计和差异置信水平的一些关键值。 如果'测试统计'小于'临界值'，我们可以拒绝原假设并说该序列是平稳的。
        # 检验结果显示，p-value=0.414, 远远大于5%的临界值，说明零假设是成立的，即序列 train 是非平稳的。
        adf_data = sts.adfuller(self.train)
        self.tagADF(adf_data)

        # In[]: 为了让时间序列平稳，需要对 train 序列做差分运算

        # df.diff 差分运算，默认是后一行减前一行
        # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html?highlight=diff#pandas.DataFrame.diff
        diff = self.train.diff(1).dropna()  # df.dropna 删除有缺失值的行
        # plt.figure(figsize=(11,6))  # 指定显示大小
        # plt.plot(diff, label='Diff')  # 绘制趋势图
        # plt.legend(loc=0)  # 显示图例，loc指定图例位置，0为最佳位置。

        # In[]:
        # 验证差分数据的平稳性，和第一次验证方法相同
        adf_Data1 = sts.adfuller(diff)
        self.tagADF(adf_Data1)  # p-value很小，零假设不成立，因此，diff数据序列符合平稳性要求。

        # In[]: 确定ARIMA的阶数p,q

        # ARMA(p,q)是AR(p)和MA(q)模型的组合，关于p和q的选择，一种方法是观察自相关图ACF和偏相关图PACF,
        # 另一种方法是通过借助AIC、BIC等统计量自动确定。
        ic = sm.tsa.arma_order_select_ic(
            self.train,
            max_ar=8,
            max_ma=8,
            ic=['aic', 'bic', 'hqic']
        )

        # In[]: 建立模型并拟合数据

        ARMAModel = sm.tsa.ARIMA(self.train, order=(4,1,2)).fit()  # order=(p,d,q)
        # ARIMA的参数中，输入数据train应该是原始数据，
        # d的含义是，输入序列需要先经过一个d阶的差分，变成一个平稳序列后才能进行数据拟合
        # ARIMA会对train先做一个差分运算，然后再拟合数据。
        # 拟合出来的数据和diff是一个数量级的，而与原数据 train之间，差一个差分还原运算。

        # fittedvalues和diff对比
        # plt.figure(figsize=(11, 6))
        # plt.plot(diff, 'r', label='Orig')
        # plt.plot(ARMAModel.fittedvalues, 'g',label='ARMA Model')
        # plt.legend()


        # 样本内预测
        predicts = ARMAModel.predict()

        # 因为预测数据是根据差分值算的，所以要对它一阶差分还原
        train_shift = self.train.shift(1)  # shift是指series往后平移1个时刻
        pred_recover = predicts.add(train_shift).dropna()  #这里add是指两列相加，按index对齐

        # 模型评价指标 1：计算 score
        delta = ARMAModel.fittedvalues - diff
        score = 1 - delta.var()/self.train.var()
        print('score:\n', score)

        # 模型评价指标 1：使用均方根误差（RMSE）来评估模型样本内拟合的好坏。
        #利用该准则进行判别时，需要剔除“非预测”数据的影响。
        #train_vs = self.train[pred_recover.index]  # 过滤没有预测的记录
        # plt.figure(figsize=(11, 6))
        # train_vs.plot(label='Original')
        # pred_recover.plot(label='Predict')
        # plt.legend(loc='best')
        # plt.title('RMSE: %.4f'% np.sqrt(sum((pred_recover-train_vs)**2)/train_vs.size))
        # plt.show()

        #预测目标(会延迟一个工作日)

        f = ARMAModel.forecast(10)
        print(f[0])
        return f[0]

