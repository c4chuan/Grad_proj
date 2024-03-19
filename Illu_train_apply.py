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
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(seed=2024)
np.random.seed(seed = 2024)

# 初始化权重频率矩阵
w_fre = [np.zeros((20,1),dtype=int),np.zeros((30,20),dtype=int),np.zeros((1,30),dtype=int)]
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
        _,num, co = coor
        x,y,z = co
        w_fre[x][y][z] += 1
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


def plot_coors(save_path,coors):
    temp_w_fre = [np.zeros((20, 1), dtype=int), np.zeros((30, 20), dtype=int), np.zeros((1, 30), dtype=int)]
    for coor in coors:
        _,num, co = coor
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
def Illustrate_train_apply(theta, s,step):
    # 定义数据存储目录
    result_path = './figures/Illu/'
    sp_path = result_path + f'theta={theta}_s={s}_step={step}'

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(sp_path):
        os.mkdir(sp_path)
    s = int(s * 650)

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
    with open(sp_path+'/result.txt','w') as file:
        while(True):
            y_pred = model(x)

            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 梯度更新后对模型进行稀疏优化
            if epoch % step == 0:
                w_1 = copy_model_parameters(model)
                coors = find_largest_coor(w_1, s)
                update_w_fre(coors)
                model.load_state_dict(sparsify_model_dict(model, coors))
                flag, total_dis = is_converge(w_0, model, theta)
                his_loss.append(np.sqrt(loss.item()))
                his_dis.append(total_dis.detach().cpu())
                file.write(f"epoch:{epoch} loss:{np.sqrt(loss.item())} dis:{total_dis}\n")
                print(f"epoch:{epoch} loss:{np.sqrt(loss.item())} dis:{total_dis}")
                w_0 = copy_model_parameters(model)
            epoch+=1

            if total_dis.detach().cpu()<=theta and epoch>1000:
                plot_w_fre(save_path=sp_path)
                plot_coors(save_path=sp_path,coors=coors)
                break
    file.close()


    #测试模型
    model.eval()
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

# 解释性实验1训练
# para_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
para_list = [0.05]
for step in [100]:
    for s in para_list:
        Illustrate_train_apply(theta=10e-5, s=s, step=step)