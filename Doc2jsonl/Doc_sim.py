import pandas as pd
import os
import textdistance

def get_corresponding_col_item(cols, df):
    dic = {}
    for col in cols:
        dic[col] = []
        for item in df[col].tolist():
            if item not in dic[col]:
                dic[col].append(item)
    return dic

def is_sim(lis_1,lis_2,thres):
    sum = 0
    for item_1 in lis_1:
        max_dis = -1
        for item_2 in lis_2:
            max_dis = max(max_dis,textdistance.jaro_winkler(str(item_1),str(item_2)))
        sum += max_dis
    return sum/len(lis_1) > thres, sum/len(lis_1)
# 选择参数
# mode = 'selected'
mode = 'all'
thres = 0.8

# 读取文件
df_1 = pd.read_csv('./ods_xttcostprocessprod.csv')
if mode == 'all':
    cols_1 = df_1.columns
else:
    cols_1 = ['PERIODNUM']

# 需匹配的Excel数据
data_path = './数据建模-维度设计v2.0'

dic_1 = get_corresponding_col_item(cols_1, df_1)
label_need = []
label_match = []
label_dim = []
sims = []
for k_1, v_1 in dic_1.items():
    label_need.append(k_1)
    match_key = ''
    last_sim = -1
    dim_name = ''
    print(k_1)
    for file in os.listdir(data_path):
        df_2 = pd.read_excel(data_path+'/'+file)
        dic_2 = get_corresponding_col_item(df_2.columns,df_2)
        for k_2,v_2 in dic_2.items():
            flag, sim = is_sim(v_1, v_2,thres)
            if flag:
                last_sim = sim
                match_key = k_2
                dim_name = file
    label_dim.append(dim_name)
    label_match.append(match_key)
    sims.append(last_sim)

df_out = pd.DataFrame()
df_out['需匹配字段'] = label_need
df_out['来源维度表'] = label_dim
df_out['匹配字段'] = label_match
df_out['相似度'] = sim
print(df_out)