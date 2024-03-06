import torch
import models.Illustrate_model as IL_model
import Data_generate
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data.dataloader
import data_loaders.TS_loader as TS_loader
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

def TS_train(epoch, batch_size):
    # 设置device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('training....')

    #生成双螺旋数据
    sample_num = 2000
    X,y = Data_generate.data_two_spiral_generate(sample_num)
    X,y = X.to(device),y.to(device)

    # 划分数据集
    split_ratio = 0.8
    dataset = TS_loader.TSDataset(X,y)
    train_data,test_data = torch.utils.data.random_split(dataset,[int(split_ratio*sample_num),sample_num-int(split_ratio*sample_num)])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 256, shuffle= True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle= True)
    # for batch_idx, (x,y) in enumerate(train_dataloader):
    #     b_x_idx = []
    #     g_x_idx = []
    #     for i in range(len(y)):
    #         if y[i] == 0:
    #             b_x_idx.append(i)
    #         else:
    #             g_x_idx.append(i)
    #     b_x = x[b_x_idx]
    #     g_x = x[g_x_idx]
    #     print(b_x)
    #     plt.scatter(b_x.cpu()[:][:,0],b_x.cpu()[:][:,1],color = 'b')
    #     plt.scatter(g_x.cpu()[:][:,0],g_x.cpu()[:][:,1],color = 'g')
    # plt.show()
    # model = IL_model.fcModel(2,2).to(device)
    model = IL_model.Net_two_spiral(2,20,30,2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for e in range(epoch):
        model.train()
        for batch_idx, (x,y) in enumerate(train_dataloader):
            preds = model(x)
            loss = criterion(preds,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss,total_acc_num = 0,0

        model.eval()
        for x,y in train_dataloader:
            preds = model(x)
            loss = criterion(preds,y).item()
            total_loss += loss
            acc_num = (preds.argmax(1) == y).type(torch.float).sum().item()
            total_acc_num += acc_num
        epoch_loss = total_loss / batch_idx
        epoch_accuracy = total_acc_num / len(train_dataloader.dataset)

        print(f'Epoch:{e},Train_loss:{epoch_loss},Train_acc:{epoch_accuracy}')






#解释性实验1训练
# Illustrate_train(10000)

#双螺旋分类实验训练
TS_train(epoch = 100,batch_size = 64)