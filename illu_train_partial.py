import time
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
import torch.nn.utils.prune as pr
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_data_num = 1000
valid_data_num = 3000
steps = [100]
para_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
theta = 10e-2
num_layer_1 = 200
num_layer_2 = 300
layer_num = 3
cuda_device = 'cuda:0'
method = 'partial' # 'partial'/'global'
partial_mask = [0,1,0]
early_stop = 5
count = 0


# 初始化权重频率矩阵
w_fre = [np.zeros((num_layer_1,1),dtype=int),np.zeros((num_layer_2,num_layer_1),dtype=int),np.zeros((1,num_layer_2),dtype=int)]

def recover_model(model):
    if layer_num == 3:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight'), (model.layer3, 'weight')]
    elif layer_num == 2:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight')]
    if method == 'global':
        for module in modules:
            pr.remove(module[0],'weight')
    elif method == 'partial':
        for i in range(len(partial_mask)):
            if partial_mask[i]:
                pr.remove(modules[i][0],'weight')

def prune_model(model,method,s):
    if layer_num == 3:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight'), (model.layer3, 'weight')]
    elif layer_num == 2:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight')]
    if method == 'global':
        pr.global_unstructured(modules, pruning_method=pr.L1Unstructured, amount =s)
    elif method == 'partial':
        for module in modules[1:-1]:
            pr.l1_unstructured(module[0],'weight',amount=s)

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
        # print(p.shape)
        if len(p.shape)>1:
            w.append(p.clone())
    return w
def update_w_fre(coors):
    for coor in coors:
        co = coor
        x,y,z = co
        w_fre[x][y][z] += 1
def find_largest_coor(model):
    '''
    找到模型参数中最大的S个参数的坐标
    :param w:
    :param s:
    :return:
    '''
    pq = []
    dic = model.state_dict()

    for i in range(layer_num):
        if f'layer{i+1}.weight_mask' in dic:
            mask = dic[f'layer{i+1}.weight_mask']
            for x in range(len(mask)):
                for y in range(len(mask[x])):
                    if mask[x][y] != 0:
                        pq.append((i,x,y))
    return pq


def plot_coors(save_path,coors):
    temp_w_fre = [np.zeros((num_layer_1, 1), dtype=int), np.zeros((num_layer_2, num_layer_1), dtype=int), np.zeros((1, num_layer_2), dtype=int)]
    for coor in coors:
        co = coor
        x,y,z = co
        temp_w_fre[x][y][z] += 1
    for i in range(len(w_fre)):
        plt.title(f'Final Largest Weights Matrix {i}')
        sns.heatmap(temp_w_fre[i],cmap='YlGnBu',linewidth=0.9,linecolor='white',
                vmax=None,vmin=None,square=True)
        plt.savefig(save_path+f'/w_{i}_final.png')
        plt.show()

def plot_w_fre(save_path):
    for i in range(len(w_fre)):
        plt.title(f'Largest Weights Frequency Matrix {i}')
        sns.heatmap(w_fre[i], cmap='YlGnBu', linewidth=0.9, linecolor='white',
                    vmax=None, vmin=None, square=True)
        plt.savefig(save_path+f'/w_{i}_fre.png')
        plt.show()
def Illustrate_train_apply(theta, s,step,seed):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    # 定义数据存储目录
    result_path = f'./figures/Illu_large_{method}_{seed}/'
    sp_path = result_path + f'theta={theta}_s={s}_step={step}'

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(sp_path):
        os.mkdir(sp_path)

    # 设置device
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 得到训练数据
    x,y = Data_generate.data_sinc_generate(num = train_data_num)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x,y = x.to(device), y.to(device)

    # 设置模型
    model = IL_model.Net_sinc(1,num_layer_1,num_layer_2,1,layer_num = layer_num)
    model.to(device)


    # 设置损失函数和优化算法
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    epoch = 0
    w_0 = copy_model_parameters(model)

    # 记录模型历史loss和前一时刻模型参数的距离
    his_loss = []
    his_dis = []
    times = []
    least_loss = 99999

    # 开始训练模型
    with open(sp_path+'/result.txt','w') as file:
        while(True):
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 梯度更新后对模型进行稀疏优化
            if epoch % step == 0:
                st_time = time.time()
                w_1 = copy_model_parameters(model)
                prune_model(model,method,s)
                coors = find_largest_coor(model)
                update_w_fre(coors)
                recover_model(model)
                flag, total_dis = is_converge(w_0, model, theta)
                train_loss = np.sqrt(loss.item())
                his_loss.append(np.sqrt(loss.item()))
                his_dis.append(total_dis.detach().cpu())

                model.eval()
                valid_x = torch.Tensor(np.linspace(-1.5 * np.pi, 1.5 * np.pi, num=valid_data_num).T).reshape(valid_data_num, 1).to(device)
                valid_y = model(valid_x)
                valid_x = valid_x.cpu().detach().numpy()
                valid_y = valid_y.cpu().detach().numpy()
                true_y = np.sinc(valid_x)
                # print(true_y.shape)
                # print(valid_y.shape)
                valid_loss = np.sqrt(((true_y-valid_y)**2).sum())

                file.write(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_loss} dis:{total_dis}\n")
                print(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_loss} dis:{total_dis}")
                w_0 = copy_model_parameters(model)

                end_time = time.time()
                times.append(end_time-st_time)

                if valid_loss < least_loss:
                    torch.save(model,sp_path+'/best_model.pth')
                    least_loss = valid_loss
                    count = 0
                else:
                    count += 1

            epoch+=1

            if count > early_stop or epoch>50000:
                # plot_w_fre(save_path=sp_path)
                # plot_coors(save_path=sp_path,coors=coors)
                total_time = 0
                for t in times:
                    total_time += t
                av_time = total_time / len(times)
                file.write(f'best_valid_loss:{least_loss} average_time:{av_time}')
                print(f'best_valid_loss:{least_loss} average_time:{av_time}')
                break

    #测试模型
    model = torch.load(sp_path+'/best_model.pth')
    model.eval()
    # print(model.state_dict())
    test_x = torch.Tensor(np.linspace(-1.5*np.pi,1.5*np.pi,num=300).T).reshape(300,1).to(device)
    test_y = model(test_x)
    test_x = test_x.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
    true_y = np.sinc(test_x)
    plt.plot(test_x,test_y)
    plt.plot(test_x,true_y,color = 'y')
    plt.savefig(sp_path+'/fit_fig.png')
    plt.show()

    # 绘制时间曲线图
    time_x = [i*step for i in range(int(epoch/step))]
    plt.plot(time_x,his_loss[1:len(time_x)+1])
    plt.savefig(sp_path+'/loss_fig.png')
    plt.show()
    plt.plot(time_x,his_dis[1:len(time_x)+1])
    plt.savefig(sp_path+'/dis_fig.png')
    plt.show()

for seed in [3,4,5,6]:
    for step in steps:
        for s in para_list:
            Illustrate_train_apply(theta=theta, s=s, step=step,seed=seed)