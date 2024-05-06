import torch
model = torch.load('done_model/'+'Transformer_model.pkl')
dic = model.state_dict()
def print_model(model):
    print(model)
    for key, value in model.state_dict().items():
        if 'encoders' in key:
            print(key, value.shape)
        if 'fc1.weight' == key:
            print(key, value.shape)
def check_zero_num(tensor):
    count = 0
    if len(tensor.shape)>1:
        for x in range(len(tensor)):
            for y in range(len(tensor[x])):
              if tensor[x][y] != 0:
                  count+=1
    else:
        return len(tensor),len(tensor)

    return count,tensor.shape[0]*tensor.shape[1]

print(model)
for key,value in dic.items():
    if not key == 'embedding.weight':
        print(f'--{key}')
        print(check_zero_num(value))
