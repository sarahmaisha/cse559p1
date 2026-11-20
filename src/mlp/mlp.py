
import torch
import torch.nn as nn


class MLP(nn.Module):
    # each cell line has 1046 gene expression values. Each layer goes from 1046 -> 128 -> 64 -> 1 
    def __init__(self, input_dim=1046, hidden1=128, hidden2=64):
        super().__init__()

        # first hidden layer
        self.l1 = nn.Linear(input_dim, hidden1)
        # second lyaer
        self.l2 = nn.Linear(hidden1, hidden2)
        # output layer
        self.l3 = nn.Linear(hidden2, 1)

        # nonlinearity so the model can learn relationships that linear regression can't
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))

        #noticed some ln_ic50 values can be negative, so no ReLu here
        x = self.l3(x)

        return x

