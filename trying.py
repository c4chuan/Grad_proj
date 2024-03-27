import torch
import numpy as np
import heapq
device = 'cuda:0'
a= torch.Tensor([[1,2,3],[4,5,6]]).to(device)
b = torch.ones((1000,1000)).to(device)
lis = []
for i in range(1000):
    for j in range(1000):
        lis.append((torch.tensor(0.000000000001).to('cuda:0'),(i,j),(j,i,'xx')))
print('into_heapify')
heapq.heapify(lis)
print(lis[-10:])
print(1+3)