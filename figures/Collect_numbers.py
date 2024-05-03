import os
import re

type = 'TS' # 'Illu_large' or 'TS'
method = 'partial' # 'global' or 'partial'
id_list = [2,3,4,5,6,7]
theta = 0.1
if type == 'Illu_large':
    step = 100
else:
    step = 1
reverse_s_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
s_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]



def result_analysis_ts(base_dir):
    train_acc_list = []
    test_acc_list = []
    for s in s_list:
        with open(base_dir + f'theta={theta}_s={s}_step={step}/result.txt', 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            train_acc = eval(re.findall(r'best_acc:(.+?) ', last_line)[0])
            test_acc = eval(re.findall(r'(?<=test_acc:).*$', last_line)[0])

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            f.close()
    with open(base_dir+'f_result.txt','w') as file:
        file.write('--train_acc_list\n')
        print('--train_acc_list')
        file.write(str(train_acc_list)+'\n')
        print(train_acc_list)
        file.write('--valid_acc_list\n')
        print('--valid_acc_list')
        file.write(str(test_acc_list)+'\n')
        print(test_acc_list)
        print('--reverse_s_list')
        print(reverse_s_list)

def result_analysis_illu(base_dir):

    test_loss_list = []
    valid_loss_list = []
    train_loss_list = []

    for s in s_list:
        with open(base_dir+f'theta={theta}_s={s}_step={step}/result.txt','r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            best_line = lines[-8]
            loss = round(1/eval(re.findall(r'best_valid_loss:(.+?) ',last_line)[0]),4)
            train_loss = round(eval(re.findall(r'train_loss:(.+?) ',best_line)[0]),4)
            valid_loss = round(eval(re.findall(r'best_valid_loss:(.+?) ',last_line)[0])/10,4)

            valid_loss_list.append(valid_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(loss)
            f.close()
    with open(base_dir+'f_result.txt','w') as file:
        file.write('--train_loss_list\n')
        print('--train_loss_list')
        file.write(str(train_loss_list)+'\n')
        print(train_loss_list)
        file.write('--valid_loss_list\n')
        print('--valid_loss_list')
        file.write(str(valid_loss_list)+'\n')
        print(valid_loss_list)
        print('--reverse_s_list')
        print(reverse_s_list)

def write_f_result(method,id_list):
    for id in id_list:
        pth = f'./{type}_{method}_{id}/'
        if type == 'Illu_large':
            result_analysis_illu(pth)
        elif type == 'TS':
            result_analysis_ts(pth)



def collect_from_result(method, id_list):
    f_train_list = [0 for i in range(len(s_list))]
    f_test_list = [0 for i in range(len(s_list))]
    for id in id_list:
        pth = f'./{type}_{method}_{id}'+'/f_result.txt'
        with open(pth,'r') as file:
            lines = file.readlines()
            train_list = eval(lines[1])
            test_list = eval(lines[3])
        for i in range(len(f_train_list)):
            f_train_list[i]+=train_list[i]
            f_test_list[i]+=test_list[i]
    for j in range(len(f_train_list)):
        f_train_list[j] = f_train_list[j]/len(id_list)
        f_test_list[j] = f_test_list[j]/len(id_list)
    print(f_train_list)
    print(f_test_list)




def process(method,id_list):
    write_f_result(method,id_list)
    collect_from_result(method,id_list)

process(method,id_list)