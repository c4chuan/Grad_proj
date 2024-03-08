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

def get_grid_points():
    x_l, x_r = -1,1
    y_l, y_r = -1,1
    xx,yy = np.meshgrid(np.linspace(x_l,x_r, 200),
                        np.linspace(y_l,y_r,200))
    x = np.c_[xx.ravel(),yy.ravel()]
    y = np.ones(shape = x.shape[0],dtype=np.int64)

    return x,y,xx,yy

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

    # 设置模型、损失函数和优化器
    # model = IL_model.fcModel(2,2).to(device)
    model = IL_model.Net_two_spiral(2,20,30,2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for e in range(epoch):

        #开始训练模型
        model.train()
        for batch_idx, (x,y) in enumerate(train_dataloader):
            preds = model(x)
            loss = criterion(preds,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss,total_acc_num = 0,0

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

        print(f'Epoch:{e},Train_loss:{epoch_loss},Train_acc:{epoch_accuracy}')

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
    print(f'test_loss:{test_loss}, test_acc:{test_acc}')
    draw_points_from_loader(test_dataloader)
    plt.show()

TS_train(epoch = 100,batch_size = 64)