import time

import torch
import models.Illustrate_model as IL_model
import Data_generate
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data.dataloader
import data_loaders.TS_loader as TS_loader
from matplotlib import  colors
import heapq
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn.utils.prune as pr

sample_num = 2000
valid_data_num = 3000
steps = [1]
para_list = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
# para_list = [0.8]
theta = 10e-2
num_layer_1 = 200
num_layer_2 = 300
layer_num = 3
cuda_device = 'cuda:7'
method = 'partial' # 'partial'/'global'
partial_mask = [0,1,0]
early_stop = 5
count = 0

# 初始化权重频率矩阵
w_fre = [np.zeros((num_layer_1,2),dtype=int),np.zeros((num_layer_2,num_layer_1),dtype=int),np.zeros((2,num_layer_2),dtype=int)]

def recover_model(model):
    if layer_num == 3:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight'), (model.layer3, 'weight')]
    elif layer_num == 2:
        modules = [(model.layer1, 'weight'), (model.layer2, 'weight')]
    if method == 'global':
        for module in modules:
            pr.remove(module,'weight')
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
def get_grid_points():
    x_l, x_r = -1,1
    y_l, y_r = -1,1
    xx,yy = np.meshgrid(np.linspace(x_l,x_r, 200),
                        np.linspace(y_l,y_r,200))
    x = np.c_[xx.ravel(),yy.ravel()]
    y = np.ones(shape = x.shape[0],dtype=np.int64)

    return x,y,xx,yy
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
def draw_points_from_loader(dataloader):
    for batch_idx, (x,y) in enumerate(dataloader):
        b_x_idx = []
        g_x_idx = []
        for i in range(len(y)):
            if y[i] == 0:
                b_x_idx.append(i)
            else:
                g_x_idx.append(i)
        b_x = x[b_x_idx]
        g_x = x[g_x_idx]
        plt.scatter(b_x.cpu()[:][:,0],b_x.cpu()[:][:,1],color = 'b')
        plt.scatter(g_x.cpu()[:][:,0],g_x.cpu()[:][:,1],color = 'g')
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
    for i in range(layer_num):
        model_dic[f'layer{i+1}.weight'] = new_weight[i]
    return model_dic
def plot_coors(save_path,coors):
    temp_w_fre = [np.zeros((num_layer_1, 2), dtype=int), np.zeros((num_layer_2, num_layer_1), dtype=int), np.zeros((2, num_layer_2), dtype=int)]
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
def TS_train(theta, s, step,seed):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    # 设置device
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('training....')

    #生成双螺旋数据
    X,y = Data_generate.data_two_spiral_generate(sample_num)
    X,y = X.to(device),y.to(device)

    # 划分数据集
    split_ratio = 0.8
    dataset = TS_loader.TSDataset(X,y)
    train_data,test_data = torch.utils.data.random_split(dataset,[int(split_ratio*sample_num),sample_num-int(split_ratio*sample_num)])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 256, shuffle= True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle= True)

    # 设置模型、损失函数和优化器
    # model = IL_model.fcModel(2,2).to(device)
    model = IL_model.Net_two_spiral(2,num_layer_1,num_layer_2,2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch = 0
    w_0 = copy_model_parameters(model)

    # 定义数据存储目录
    result_path = f'./figures/TS_{method}_{seed}/'
    sp_path = result_path + f'theta={theta}_s={s}_step={step}'

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(sp_path):
        os.mkdir(sp_path)
    s = int(s * 700)

    # 记录模型历史loss,acc和前一时刻模型参数的距离
    his_loss = []
    his_acc = []
    his_dis = []
    times = []
    count = 0
    best_acc = -1

    # 开始训练模型
    with open(sp_path + '/result.txt', 'w') as file:
        while True:
            #开始训练模型
            model.train()
            for batch_idx, (x,y) in enumerate(train_dataloader):
                preds = model(x)
                loss = criterion(preds,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            total_loss,total_acc_num = 0,0

            # 梯度更新后对模型进行稀疏优化
            if epoch % step == 0:
                st_time = time.time()
                w_1 = copy_model_parameters(model)
                prune_model(model,method,s)
                coors = find_largest_coor(model)
                update_w_fre(coors)
                recover_model(model)
                flag, total_dis = is_converge(w_0, model, theta)
                w_0 = copy_model_parameters(model)
                end_time = time.time()
                times.append(end_time-st_time)

            # 计算训练过程中的训练损失和正确率
            model.eval()
            for x,y in train_dataloader:
                preds = model(x)
                loss = criterion(preds,y).item()
                total_loss += loss
                acc_num = (preds.argmax(1) == y).type(torch.float).sum().item()
                total_acc_num += acc_num
            epoch_loss = total_loss / batch_idx
            epoch_accuracy = total_acc_num / len(train_dataloader.dataset)
            his_loss.append(epoch_loss)
            his_acc.append(epoch_accuracy)
            his_dis.append(total_dis.cpu().detach().numpy())
            file.write(f'Epoch:{epoch},Train_loss:{epoch_loss},Train_acc:{epoch_accuracy},Dis:{total_dis}\n')
            print(f'Epoch:{epoch},Train_loss:{epoch_loss},Train_acc:{epoch_accuracy},Dis:{total_dis}')

            if epoch_accuracy >best_acc:
                torch.save(model, sp_path + '/best_model.pth')
                best_acc = epoch_accuracy
                count = 0
            else:
                count += 1

            epoch+=1

            if count > early_stop or epoch > 5000:
                # plot_w_fre(save_path=sp_path)
                # plot_coors(save_path=sp_path,coors=coors)
                total_time = 0
                for t in times:
                    total_time += t
                av_time = total_time / len(times)
                file.write(f'best_acc:{best_acc} average_time:{av_time}')
                break

        x, _, xx, yy = get_grid_points()
        x = torch.Tensor(x).to(device)
        z_preds = model(x).cpu().detach().numpy()
        z_preds = np.argmax(z_preds, axis=1)
        colormap = colors.ListedColormap([ "#0877bd","#e8eaeb","#f59322"])
        plt.contourf(xx, yy, z_preds.reshape(xx.shape), cmap=colormap, alpha=0.4)

        total_test_loss, total_test_acc_num = 0,0
        for batch_idx,(x, y) in enumerate(test_dataloader):
            preds = model(x)
            loss = criterion(preds, y).item()
            total_test_loss += loss
            acc_num = (preds.argmax(1) == y).type(torch.float).sum().item()
            total_test_acc_num += acc_num
        test_loss = total_test_loss / batch_idx
        test_acc = total_test_acc_num / len(test_dataloader.dataset)
        file.write(f'test_loss:{test_loss}, test_acc:{test_acc}')
        print(f'test_loss:{test_loss}, test_acc:{test_acc}')
        draw_points_from_loader(test_dataloader)
        plt.savefig(sp_path+'/test_fig.png')
        plt.show()
        file.close()

    # 绘制时间曲线图
    time_x = [i * step for i in range(int(epoch / step)-1)]
    plt.plot(time_x, his_loss[1:len(time_x) + 1])
    plt.savefig(sp_path + '/loss_fig.png')
    plt.show()
    plt.plot(time_x, his_dis[1:len(time_x) + 1])
    plt.savefig(sp_path + '/dis_fig.png')
    plt.show()
    plt.plot(time_x, his_acc[1:len(time_x) + 1])
    plt.savefig(sp_path + '/acc_fig.png')
    plt.show()

for seed in [3,4,5,6]:
    for s in para_list:
        for step in steps:
            TS_train(theta=theta, s = s, step=step,seed=seed)