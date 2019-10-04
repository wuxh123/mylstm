# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import talib

data = np.loadtxt('300188.csv', skiprows=1, delimiter=",", usecols=(3,4,5,6,7,8,9,10,11,12))
# a = talib.CDL2CROWS(data[:,3],data[:,1],data[:,2],data[:,0])
# print(a)

# b = talib.CDL3BLACKCROWS(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)

# b = talib.CDL3INSIDE(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)

# b = talib.CDL3LINESTRIKE(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)

# b = talib.CDL3OUTSIDE(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)

# b=talib.CDL3STARSINSOUTH(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)

# b=talib.CDLBELTHOLD(data[:,3],data[:,1],data[:,2],data[:,0])
# print(b)
