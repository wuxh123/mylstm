# from flask import Flask,request
# import requests
# from flask import render_template
# import json
# from .data import getdata
# from .predict import LstmPred
# from .arma import clsArma
#
# app = Flask(__name__)
# @app.route('/cal',methods=['GET'])
# def search_form():
#     return render_template('search.html')
#
# #如果访问的是/search页面的post请求，则调用send_post（）方法
# @app.route('/cal',methods=['POST'])
# def search():
#     code = request.form['code'].strip()  # strip去除前后空格
#     print(code)
#     if code.isdigit() and len(code) == 6:
#         return send_post(code)
#     else:
#         return render_template('search.html', tips="请输入正确的姓名或编码")
#
# #如果访问的是/search页面的post请求，则调用send_post（）方法
# @app.route('/cal_1',methods=['POST'])
# def search1():
#     code = request.form['code'].strip()  # strip去除前后空格
#     print(code)
#     if code.isdigit() and len(code) == 6:
#         return send_post1(code)
#     else:
#         return render_template('search.html', tips_1="请输入正确的姓名或编码")
#
# #计算结果
# def send_post(code):
#     getdata(code)
#     print(code)
#     lstm = LstmPred()
#     pred=lstm.predictonce(code)
#     return render_template('search.html',predict='预测一：' + str(round(pred[0],2))  )
#
# #计算结果
# def send_post1(code):
#     arma = clsArma(code)
#     pred2 = arma.predict()
#
#     return render_template('search.html',predict_1='预测二(下2个交易日)：'+ str(round(pred2[0],2))+","+ str(round(pred2[1],2)) )
#
# if __name__ == '__main__':
#     app.run(
#         host='172.29.140.58',
#         port=8086,
#         debug=True
#     )