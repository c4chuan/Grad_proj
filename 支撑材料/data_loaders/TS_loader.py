import torch
from torch.utils.data import Dataset
class TSDataset(Dataset):
    def __init__(self,x,y):
        self.points = x
        self.labels = y.type(torch.long)
        self.len = len(self.labels)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx]

        return point,label

    def __len__(self):
        return self.len