import torch
import models.Illustrate_model as IL_model
from queue import PriorityQueue as PQ
import heapq
import Data_generate
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data.dataloader
import data_loaders.TS_loader as TS_loader
from matplotlib import  colors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def is_converge(w_0,w_1,theta):
    if torch.square(w_0-w_1).sum()< theta:
        return False
    else:
        return True

def copy_model_parameters(model):
    w = []
    for p in model.parameters():
        if len(p.shape)>1:
            w.append(p.detach())
    return w

def find_largest_coor(w,s):
    pq = []
    heapq.heapify(pq)
    for i in range(s):
        heapq.heappush(pq,(-999,(-999,-999,-999)))
    for x in range(len(w)):
        for y in range(len(w[x])):
            for z in range(len(w[x][y])):
                if torch.abs(w[x][y][z]) > pq[0][0]:
                    heapq.heapreplace(pq,(torch.abs(w[x][y][z]),(x,y,z)))
    return pq

def sparsify_model_dict(model, coors):
    model_dic = model.state_dict()
    new_weight = []
    for i in range(3):
        shape = model_dic[f'layer{i+1}.weight'].shape
        new_weight.append(torch.zeros(shape))
    for coor in coors:
        num, co = coor
        x,y,z = co
        new_weight[x][y][z] = num
    for i in range(3):
        model_dic[f'layer{i+1}.weight'] = new_weight[i]
    return model_dic


def Illustrate_train_apply(theta, s):
    #设置device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #得到训练数据
    x,y = Data_generate.data_sinc_generate(num = 1000)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x,y = x.to(device), y.to(device)

    #设置模型
    model = IL_model.Net_sinc(1,20,30,1)
    model.to(device)
    model.train()

    #设置损失函数和优化算法
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    w_0 = copy_model_parameters(model)


    epoch = 0

    #开始训练模型
    while(True):
        y_pred = model(x)

        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        coors = find_largest_coor(w_0, s)
        model.load_state_dict(sparsify_model_dict(model, coors))

        if epoch % 100 == 0:
            print(f"epoch:{epoch} loss:{np.sqrt(loss.item())}")
        epoch+=1


    #
    # #测试模型
    # model.eval()
    # test_x = torch.Tensor(np.linspace(-1.5*np.pi,1.5*np.pi,num=300).T).reshape(300,1).to(device)
    # test_y = model(test_x)
    # test_x = test_x.cpu().detach().numpy()
    # test_y = test_y.cpu().detach().numpy()
    # true_y = np.sinc(test_x)
    # plt.plot(test_x,test_y)
    # plt.plot(test_x,true_y,color = 'y')
    # plt.show()

# 解释性实验1训练
Illustrate_train_apply(theta = 0.01, s = 30)