import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import math

def data_sinc_generate(num = 300):
    x = np.random.uniform(-1.5*np.pi, 1.5*np.pi,[num,1])
    noise = np.random.normal(0,np.sqrt(0.005),[num,1])
    y = np.sinc(x) + noise
    plt.scatter(x, y, color='b', marker='.')
    plt.show()
    return x,y

def data_two_spiral_generate(totalnum = 2000):
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)

    N = int(totalnum/2)  # 每类样本的数量
    D = 2  # 每个样本的特征维度
    C = 2  # 样本的类别
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X = torch.zeros(N * C, D).to(device)
    Y = torch.zeros(N * C, dtype=torch.long).to(device)
    for c in range(C):
        index = 0
        t = torch.linspace(0, 1, N)  # 在[0，1]间均匀的取10000个数，赋给t
        # 下面的代码不用理解太多，总之是根据公式计算出三类样本（可以构成螺旋形）
        # torch.randn(N) 是得到 N 个均值为0，方差为 1 的一组随机数，注意要和 rand 区分开
        inner_var = torch.linspace((2 * math.pi / C) * c, (2 * math.pi / C) * (2 + c), N) + torch.randn(N) * 0.2

        # 每个样本的(x,y)坐标都保存在 X 里
        # Y 里存储的是样本的类别，分别为 [0, 1, 2]
        for ix in range(N * c, N * (c + 1)):
            X[ix] = t[index] * torch.FloatTensor((math.sin(inner_var[index]), math.cos(inner_var[index])))
            Y[ix] = c
            index += 1

    print("Shapes:")
    print("X:", X.size())
    print("Y:", Y.size())
    class_1_x = []
    plt.scatter(X.cpu()[:999][:,0],X.cpu()[:999][:,1],color='b',marker='.')
    plt.scatter(X.cpu()[1000:][:,0],X.cpu()[1000:][:,1],color='g',marker='.')
    plt.show()
    return X,Y



