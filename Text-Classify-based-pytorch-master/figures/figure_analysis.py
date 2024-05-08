import re
import matplotlib.pyplot as plt
test_acc_list = [0.6663028]
train_acc_list = [0.717]
s_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99]
reverse_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.04,0.03,0.02,0.01]
epoch_list = []
method = 'global'
for i in range(len(s_list)):
    s = s_list[i]
    result_path = f'./s={s}_theta=0.1_method={method}/results.txt'
    with open(result_path,'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        best_line = lines[-15]
        epoch_list.append((len(lines)-1)/2)
        train_acc = eval(re.findall(r'(?<=Train Accuracy:).*$', best_line)[0])
        test_acc = eval(re.findall(r'(?<=best_acc=).*$', last_line)[0])
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        file.close()
print('--reverse_list')
print(reverse_list)
print('--train_acc')
print(train_acc_list)
print('--test_acc')
print(test_acc_list)
print(epoch_list)
av = 0
for e in epoch_list:
    av += e
print(av/len(epoch_list))
# plt.plot(reverse_list,test_acc_list)
# plt.plot(reverse_list,train_acc_list)
# plt.title('s_test_acc_fig')
# plt.xlabel('parameters_percentage')
# plt.ylabel('test_acc')
# plt.savefig('s_acc.png')
# plt.show()

