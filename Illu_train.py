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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def Illustrate_train(epoch):
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

    #开始训练模型
    for e in range(epoch):
        y_pred = model(x)

        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if e % 1000 == 0:
            print(f"epoch:{e} loss:{np.sqrt(loss.item())}")

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

# 解释性实验1训练
Illustrate_train(10000)