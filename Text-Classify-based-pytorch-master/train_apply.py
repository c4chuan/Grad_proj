import os.path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import DataSet
import numpy as np
import heapq
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from Config import *
from model.TextCNN import TextCNN
from model.TextRCNN import TextRCNN
from model.TextRNN import TextRNN
from model.TextRNN_Attention import TextRNN_Attention
from model.Transformer import Transformer

def is_converge(w_0,model,theta):
    '''
    :param w_0: 前一时刻记录的模型参数
    :param model: 模型
    :param theta: 终止条件
    :return:
    '''
    w_1,_,_ = copy_model_parameters(model)
    total_dis = 0
    for i in w_0.keys():
        # print(w_0[i]-w_1[i])
        total_dis += torch.square(w_0[i]-w_1[i]).sum()
    if total_dis < theta:
        return True, total_dis
    else:
        return False,total_dis

def sparsify_model_dict(model, coors, layer_keys):
    '''
    更新模型参数，使其稀疏
    :param model:
    :param coors:
    :return:
    '''
    model_dic = model.state_dict()
    new_weight = {}
    for layer_key in layer_keys:
        shape = model_dic[layer_key].shape
        new_weight[layer_key] = torch.zeros(shape)
    for coor in coors:
        _,num, co = coor
        x,y,z = co
        new_weight[x][y][z] = num
    for layer_key in layer_keys:
        model_dic[layer_key] = new_weight[layer_key]
    return model_dic
def find_largest_coor(w,s):
    '''
    找到模型参数中最大的S个参数的坐标
    :param w:
    :param s:
    :return:
    '''
    weight_lis = []
    for x in w.keys():
        for y in range(len(w[x])):
            for z in range(len(w[x][y])):
                weight_lis.append((torch.abs(w[x][y][z].cpu()),w[x][y][z],(x,y,z)))
    # weight_lis = sorted(weight_lis,key=lambda s:s[0])
    heapq.heapify(weight_lis)
    return weight_lis[-s:]
def print_model(model):
    print(model)
    for key, value in model.state_dict().items():
        if 'encoders' in key:
            print(key, value.shape)
        if 'fc1.weight' == key:
            print(key, value.shape)
def copy_model_parameters(model):
    '''
    复制模型矩阵参数（除偏置）
    :param model:
    :return:
    '''
    w = {}
    sum = 0
    keys = []
    for key, value in model.state_dict().items():
        if 'encoders' in key and len(value.shape) > 1:
            sum += value.shape[0]*value.shape[1]
            w[key] = value.clone()
            keys.append(key)
        elif 'fc1.weight' == key:
            sum += value.shape[0] * value.shape[1]
            w[key] = value.clone()
            keys.append(key)
    return w,sum,keys
def test_model(test_iter, name, device):
    model = torch.load('done_model/'+name+'_model.pkl')
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        loss = F.cross_entropy(out, target)
        total_loss += loss.item()
        accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=label_list, digits=3))

def train_model(train_iter, dev_iter, model, name, device, step,s,theta):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.6)
    # 复制模型的参数作为t0时刻
    w_0,para_num,layer_keys = copy_model_parameters(model)
    his_dis = []
    his_train_acc = []
    his_dev_acc = []
    # print_model(model)
    model.train()
    best_acc = 0
    print('training...')
    epoch = 0
    path = './figures'
    save_path = path + f'/s={s}_theta={theta}'
    s = int(s * para_num)
    early_stop = 0
    pre_acc = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(save_path+'/results.txt','w') as file:
        while True:
            model.train()
            total_loss = 0.0
            accuracy = 0
            total_train_num = len(train_iter.dataset)
            progress_bar = tqdm(enumerate(train_iter), total=len(train_iter))
            for i,batch in progress_bar:
                feature = batch.text
                target = batch.label
                with torch.no_grad():
                    feature = torch.t(feature)
                feature, target = feature.to(device), target.to(device)
                optimizer.zero_grad()
                logit = model(feature)
                loss = F.cross_entropy(logit, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                accuracy += (torch.argmax(logit, dim=1) == target).sum().item()
                progress_bar.set_description(
                f'loss: {loss.item():.3f}')

            # 对模型进行稀疏优化
            w_1, _, _ = copy_model_parameters(model)
            coors = find_largest_coor(w_1, s)
            model.load_state_dict(sparsify_model_dict(model, coors, layer_keys))
            flag, total_dis = is_converge(w_0, model, theta)
            his_dis.append(total_dis.cpu())
            his_train_acc.append(accuracy/total_train_num)
            w_0,_ ,_ = copy_model_parameters(model)
            file.write('>>> Epoch_{}, Train loss is {}, Train Accuracy:{} Dis:{}\n'.format(epoch,total_loss/total_train_num, accuracy/total_train_num, total_dis))
            print('>>> Epoch_{}, Train loss is {}, Train Accuracy:{} Dis:{}\n'.format(epoch,total_loss/total_train_num, accuracy/total_train_num, total_dis))

            model.eval()
            total_loss = 0.0
            accuracy = 0
            total_valid_num = len(dev_iter.dataset)
            progress_bar = tqdm(enumerate(dev_iter), total=len(dev_iter))
            for i, batch in progress_bar:
                feature = batch.text  # (W,N) (N)
                target = batch.label
                with torch.no_grad():
                    feature = torch.t(feature)
                feature, target = feature.to(device), target.to(device)
                out = model(feature)
                loss = F.cross_entropy(out, target)
                total_loss += loss.item()
                accuracy += (torch.argmax(out, dim=1)==target).sum().item()

            his_dev_acc.append(accuracy/total_valid_num)
            file.write('>>> Epoch_{}, Valid loss:{}, Valid Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))
            print('>>> Epoch_{}, Valid loss:{}, Valid Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))

            if(accuracy/total_valid_num > best_acc):
                print('save model...')
                best_acc = accuracy/total_valid_num
                saveModel(model, name=name)

            if accuracy/total_valid_num > pre_acc:
                pre_acc = accuracy/total_valid_num
                early_stop = 0
            else:
                early_stop += 1

            epoch += 1
            # 设置终止条件
            if total_dis < theta or epoch > 50 or early_stop > 5:
                epoch_x = range(len(his_dis))
                plt.plot(epoch_x, his_dis)
                plt.title('dis_fig')
                plt.savefig(save_path+'/disfig.png')
                plt.show()
                plt.plot(epoch_x, his_train_acc)
                plt.title('train_acc_fig')
                plt.savefig(save_path+'/train_acc.png')
                plt.show()
                plt.plot(epoch_x, his_dev_acc)
                plt.title('dev_acc_fig')
                plt.savefig(save_path+'/dev_acc.png')
                plt.show()
                file.write(f'best_acc={pre_acc}')
                break

        file.close()

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')

name = 'Transformer'
model = Transformer()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(device)
train_iter, val_iter, test_iter = DataSet.getIter()

if __name__ == '__main__':
    for s in [0.7,0.5,0.3]:
        train_model(train_iter, val_iter, model, name, device,10,s,10e-2)
    test_model(test_iter, name, device)