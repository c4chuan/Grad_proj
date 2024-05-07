import re
import matplotlib.pyplot as plt
test_acc = []
s_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]


for s in s_list:
    result_path = f'./s={s}_theta=0.1/results.txt'
    with open(result_path,'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        acc = eval(re.findall(r'(?<=best_acc=).*$', last_line)[0])
        print(acc)
        test_acc.append(acc)
        file.close()
s_list.append(1)
test_acc.append(0.6663028)

plt.plot(s_list,test_acc)
plt.title('s_test_acc_fig')
plt.xlabel('parameters_percentage')
plt.ylabel('test_acc')
plt.savefig('s_acc.png')
plt.show()