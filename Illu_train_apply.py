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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(seed=2024)
np.random.seed(seed = 2024)
def is_converge(w_0,model,theta):
    '''
    :param w_0: 前一时刻记录的模型参数
    :param model: 模型
    :param theta: 终止条件
    :return:
    '''
    w_1 = copy_model_parameters(model)
    total_dis = 0
    for i in range(len(w_0)):
        # print(w_0[i]-w_1[i])
        total_dis += torch.square(w_0[i]-w_1[i]).sum()
    if total_dis < theta:
        return True, total_dis
    else:
        return False,total_dis

def copy_model_parameters(model):
    '''
    复制模型矩阵参数（除偏置）
    :param model:
    :return:
    '''
    w = []
    for p in model.parameters():
        if len(p.shape)>1:
            w.append(p.clone())
    return w

def find_largest_coor(w,s):
    '''
    找到模型参数中最大的S个参数的坐标
    :param w:
    :param s:
    :return:
    '''
    pq = []
    tensor_list = torch.full((s,1),-1)
    heapq.heapify(pq)
    for i in range(s):
        heapq.heappush(pq,(tensor_list[i],-1,(-999,-999,-999)))
    for x in range(len(w)):
        for y in range(len(w[x])):
            for z in range(len(w[x][y])):
                if torch.abs(w[x][y][z].cpu()) > pq[0][0]:
                    heapq.heapreplace(pq,(torch.abs(w[x][y][z].cpu()),w[x][y][z],(x,y,z)))
    return pq

def sparsify_model_dict(model, coors):
    '''
    更新模型参数，使其稀疏
    :param model:
    :param coors:
    :return:
    '''
    model_dic = model.state_dict()
    new_weight = []
    for i in range(3):
        shape = model_dic[f'layer{i+1}.weight'].shape
        new_weight.append(torch.zeros(shape))
    for coor in coors:
        _,num, co = coor
        x,y,z = co
        new_weight[x][y][z] = num
    for i in range(3):
        model_dic[f'layer{i+1}.weight'] = new_weight[i]
    return model_dic


def Illustrate_train_apply(theta, s):

    # 设置device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 得到训练数据
    x,y = Data_generate.data_sinc_generate(num = 1000)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x,y = x.to(device), y.to(device)

    # 设置模型
    model = IL_model.Net_sinc(1,20,30,1)
    model.to(device)
    model.train()

    # 设置损失函数和优化算法
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    epoch = 0
    w_0 = copy_model_parameters(model)

    # 记录模型历史loss和前一时刻模型参数的距离
    his_loss = []
    his_dis = []

    # 开始训练模型
    while(True):
        # 首先记录上一时刻模型的参数w_0

        y_pred = model(x)

        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 梯度更新后对模型进行稀疏优化

        if epoch % 100 == 0:
            w_1 = copy_model_parameters(model)
            coors = find_largest_coor(w_1, s)
            model.load_state_dict(sparsify_model_dict(model, coors))
            flag, total_dis = is_converge(w_0, model, theta)
            his_loss.append(np.sqrt(loss.item()))
            his_dis.append(total_dis.detach().cpu())
            print(f"epoch:{epoch} loss:{np.sqrt(loss.item())} dis:{total_dis}")
            w_0 = copy_model_parameters(model)
        epoch+=1

        if total_dis.detach().cpu()<=theta and epoch>1000:

            break


    #测试模型
    model.eval()
    test_x = torch.Tensor(np.linspace(-1.5*np.pi,1.5*np.pi,num=300).T).reshape(300,1).to(device)
    test_y = model(test_x)
    test_x = test_x.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
    true_y = np.sinc(test_x)
    plt.plot(test_x,test_y)
    plt.plot(test_x,true_y,color = 'y')
    plt.show()

    # 绘制时间曲线图
    time_x = [i*100 for i in range(int(epoch/100))]
    plt.plot(time_x,his_loss[1:len(time_x)+1])
    plt.show()
    plt.plot(time_x,his_dis[1:len(time_x)+1])
    plt.show()

# 解释性实验1训练
Illustrate_train_apply(theta = 10e-4, s = int(0.2*650))