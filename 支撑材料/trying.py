import time
import torch
import heapq
list = []

for i in range(1800000):
    a = torch.Tensor([0.1]).to('cuda:2')
    list.append(a)
# for i in range(450000):
#     a = torch.Tensor([0.1]).to('cuda:2')
#     list.append(a)
st_time = time.time()
heapq.heapify(list)
end_time = time.time()
list = []
print(end_time-st_time)