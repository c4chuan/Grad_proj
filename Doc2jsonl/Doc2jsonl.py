import pandas as pd
import os
path_1 = './数据建模-事实表设计 V2.0.xlsx'
path_2 = './数据建模-维度设计v2.0'

def read_multi_sheet(path):
    with open('./维度设计.txt','w') as wf :
        for file in os.listdir(path):
            df = pd.read_excel(path+f'/{file}')
            keys = None
            for row in df.itertuples():
                if row[0] == 0:
                    keys = row[1:]
                else:
                    dic = {'表名':file.replace('.xlsx','')}
                    for i in range(1,len(row)):
                        dic[keys[i-1]] = row[i]
                    wf.write(str(dic)+'\n')
def read_multi_sheet_name_only(path):
    with open('./维度设计_表+字段_only.txt','w') as wf :
        for file in os.listdir(path):
            df = pd.read_excel(path+f'/{file}')
            keys = None
            dic = {'表名': file.replace('.xlsx', '')}
            for row in df.itertuples():
                if row[0] == 0:
                    keys = row[1:]
                    dic['字段'] = keys
                    wf.write(str(dic)+'\n')
                else:
                    break
def read_single_sheet(path):
    with open('./事实表设计.txt','w') as wf :
        df = pd.read_excel('./数据建模-事实表设计 V2.0.xlsx')
        cols = df.columns
        for row in df.itertuples():
            dic = {}
            for i in range(1, len(row)):
                dic[cols[i - 1]] = row[i]
            print(dic)
            wf.write(str(dic) + '\n')

read_multi_sheet_name_only(path_2)
# read_single_sheet(path_1)