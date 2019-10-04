# -*- coding: utf-8 -*-
import datetime
import json
from sys import argv

import requests

# try:
#     r = 2051 #int(argv[1])
#     if not r:
#         print('股票代码错误，请确认格式为"300270"')
# except:
#     print('股票代码错误，请确认格式为"300270"')
#     exit()
# code = '002051'#'#argv[1]
def getdata(code):
    start = '20190101'
    end = str(datetime.datetime.now().date()).replace('-', '')
    url = 'http://quotes.money.163.com/hs/service/diyrank.php?query=SYMBOL%3A{code}'.format(code=code)
    r = requests.get(url)
    code2 = json.loads(r.text)['list'][0]['CODE']
    url = 'http://quotes.money.163.com/service/chddata.html?code={code}&start={start}&end={end}&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'.format(
        code=code2, start=start, end=end)
    r = requests.get(url)
    with open('{code}.csv'.format(code=code), 'w') as f:
        lst = r.text.split('\r')
        f.write(lst[0])
        for i in reversed(lst[1:]):
            if 'None' not in i and len(i)>10:
                f.write(i.replace("'",""))
        # if 'None' in r.text:
        #     f.write(r.text)
    print('{code} 获取数据成功'.format(code=code))

getdata('600649')