import os
import re
base_dir = './result'

file_names = os.listdir(base_dir)
def process(word):
    preps = ['之','的']
    verbs = ['指','即','为']
    sp_word = word
    for prep in preps:
        if prep in word:
            sp_word = re.findall(f'(?<={prep}).*$', word)[0]
    for v in verbs:
        if v in sp_word:
            sp_word = re.findall(f'(?<={v}).*$', word)[0]
    return sp_word
def splitness(words):
    new_words = []
    split_nodes = ['、','或','与']
    for word in words:
        flag = False
        for node in split_nodes:
            if node in word:
                new_words += word.split(node)
                flag = True
        if not flag:
            new_words.append(word)
    final_words = []
    for word in new_words:
        if len(word)<=3 and len(word)>0 and '丨' not in word and '】' not in word:
            final_words.append(word)
    return final_words

word_ex_dic = {}
ex_fre_dic = {}
for file_name in file_names:
    print(file_name)
    file_path = base_dir+f'/{file_name}'
    with open(file_path,'r') as file:
        lines = file.readlines()
        for line in lines:
            if '【' in line and '】' in line:
                word = re.findall('【.*?】',line)[0][1:-1]
                if '。' not in line:
                    explain = re.findall("(?<=[指喻即称一]).*$", line)
                else:
                    explain = re.findall("[指喻即称].*?。",line)
                    explain = [i[1:-1] for i in explain]
                explain = [process(per_word) for per_word in explain]
                explain = splitness(explain)
                # 加入词典中
                word_ex_dic[word] = explain
                for ex in explain:
                    if ex not in ex_fre_dic:
                        ex_fre_dic[ex] = 1
                    else:
                        ex_fre_dic[ex] += 1

del_lis = []
for k,v in ex_fre_dic.items():
    # print(v)
    if v<=2:
        del_lis.append(k)
for k in del_lis:
    ex_fre_dic.pop(k)

print(ex_fre_dic.keys())
print(len(ex_fre_dic.keys()))