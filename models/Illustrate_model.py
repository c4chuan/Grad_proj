import torch
from torch import nn

class Net_sinc(nn.Module):

    def  __init__(self,n_input,n_hidden_1,n_hidden_2,n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_input,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,n_out)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return self.layer3(x)

class Net_two_spiral(nn.Module):

    def  __init__(self,n_input,n_hidden_1,n_hidden_2,n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_input,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,n_out)
        self.ReLU = nn.ReLU

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x


class fcModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        """Model class constructor"""
        super(fcModel, self).__init__()

        self.linear = nn.Linear(input_shape, 8)
        self.fully_connected_stack = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, output_shape),
        )

    def forward(self, x):
        x = self.linear(x)
        logits = self.fully_connected_stack(x)
        return logits
