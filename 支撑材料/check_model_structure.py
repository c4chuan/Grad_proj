import torch

model = torch.load('./figures/Illu_large_global_3/theta=0.1_s=0.4_step=100/best_model.pth',map_location='cuda:0')
print(model.layer3.weight)