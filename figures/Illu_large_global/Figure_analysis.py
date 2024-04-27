import pandas as pd
import os
import re
import matplotlib.pyplot as plt

base_dir = './'
comb = []
def get_paras(paths):
    for path in paths:
        if not path == 'Figure_analysis.py':
            theta = eval(re.findall(r'theta=(.+?)_', path)[0])
            s = eval(re.findall(r's=(.+?)_', path)[0])
            step = eval(re.findall(r'(?<=step=).*$', path)[0])
            comb.append((theta, s, step))

def plot_s_loss_curve():
    theta = 0.001
    step = 100
    s_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    loss_list = []

    for s in s_list:
        with open(base_dir+f'theta={theta}_s={s}_step={step}/result.txt','r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            loss = round(1/eval(re.findall(r'loss:(.+?) ',last_line)[0]),4)
            loss_list.append(loss)
            f.close()

    plt.plot(s_list,loss_list)
    plt.title('s_1/loss_curve')
    plt.xlabel('parameters_percentage')
    plt.ylabel('1/loss')
    plt.savefig('s_performance_curve.png')
    plt.show()

def plot_s_epoch_curve():
    theta = 0.001
    step = 100
    s_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    epoch_list = []

    for s in s_list:
        with open(base_dir + f'theta={theta}_s={s}_step={step}/result.txt', 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            epoch = eval(re.findall(r'epoch:(.+?) ', last_line)[0])
            epoch_list.append(epoch)
            f.close()

    plt.plot(s_list, epoch_list)
    plt.title('s_epoch_curve')
    plt.xlabel('parameters_percentage')
    plt.ylabel('final_epoch')
    plt.savefig('s_epoch_curve.png')
    plt.show()
plot_s_loss_curve()
plot_s_epoch_curve()