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
import torch.nn.utils.prune as pr

from Config import *
from model.TextCNN import TextCNN
from model.TextRCNN import TextRCNN
from model.TextRNN import TextRNN
from model.TextRNN_Attention import TextRNN_Attention
from model.Transformer import Transformer

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

def print_model(model):
    print(model)
    for key, value in model.state_dict().items():
        if 'encoders' in key:
            print(key, value.shape)
        if 'fc1.weight' == key:
            print(key, value.shape)

def get_modules(model):
    modules = []
    for encoder in model.encoders:
        modules.append(encoder.attention.fc_Q)
        modules.append(encoder.attention.fc_K)
        modules.append(encoder.attention.fc_V)
        modules.append(encoder.attention.fc)
        modules.append(encoder.feed_forward.fc1)
        modules.append(encoder.feed_forward.fc2)
    modules.append(model.fc1)
    return modules
def prune_model(model,s,method):
    prune_modules = get_modules(model)
    if method == 'partial':
        for m in prune_modules:
            pr.l1_unstructured(m,'weight',amount=s)
    elif method == 'global':
        prune_modules = [(m,'weight') for m in prune_modules]
        pr.global_unstructured(prune_modules, pruning_method=pr.L1Unstructured, amount =s)

def recover_model(model):
    prune_modules = get_modules(model)
    for m in prune_modules:
        pr.remove(m,'weight')

def train_model(train_iter, dev_iter, model, name, device, step,s,theta):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.6)
    # 复制模型的参数作为t0时刻
    his_dis = []
    his_train_acc = []
    his_dev_acc = []
    # print_model(model)
    model.train()
    best_acc = 0
    print('training...')
    epoch = 0
    path = './figures'
    save_path = path + f'/s={s}_theta={theta}_method={method}'
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
            prune_model(model,s,method)
            recover_model(model)
            his_train_acc.append(accuracy/total_train_num)

            file.write('>>> Epoch_{}, Train loss is {}, Train Accuracy:{}\n'.format(epoch,total_loss/total_train_num, accuracy/total_train_num))
            print('>>> Epoch_{}, Train loss is {}, Train Accuracy:{}\n'.format(epoch,total_loss/total_train_num, accuracy/total_train_num))

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
            if epoch > 50 or early_stop > 5:
                epoch_x = range(len(his_train_acc))
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

method = 'global' # global / partial
step = 10
theta = 10e-2
name = 'Transformer'
model = Transformer()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = DataSet.getIter()
print_model(model)


if __name__ == '__main__':
    for s in [0.99]:
        train_model(train_iter, val_iter, model, name, device,step,s,theta)
    test_model(test_iter, name, device)