from django.http import HttpResponse
from django.shortcuts import render
import requests
from flask import render_template
import json
from .process.data import getdata
from .process.predict import LstmPred
from .process.arma import clsArma
def home(request):
    context = {}
    context['predict'] = ''
    context['tips'] = ''
    return render(request, 'search.html', context)

def cal(request):
    code = request.POST.get('code').strip()  # strip去除前后空格
    getdata(code)
    print(code)
    lstm = LstmPred()
    pred = lstm.predictonce(code)
    context = {}
    context['predict'] = '预测一：' + str(round(pred[0], 2))
    context['tips'] = ''
    return render(request, 'search.html', context)

def cal_1(request):
    code = request.POST.get('code').strip()  # strip去除前后空格
    arma = clsArma(code)
    pred2 = arma.predict()
    context = {}
    context['predict_1'] = '预测二(下2个交易日)：' + str(round(pred2[0], 2)) + "," + str(round(pred2[1], 2))
    context['tips_1'] = ''
    return render(request, 'search.html', context)